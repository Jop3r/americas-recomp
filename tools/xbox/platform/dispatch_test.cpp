// dispatch_test: run a lowered Box32/Box16 rehydration shader (cbv b0, raw uav u0 = compressedBVH, raw uav u1 = decompressedBVH) on the
// GPU and on WARP with the same pseudo-random input, then compare the two outputs.
// build: cl /nologo /EHsc /O2 /std:c++17 dispatch_test.cpp d3d12.lib dxgi.lib /Fe:dispatch_test.exe
// usage: dispatch_test shader.dxil [groups] [seed]        (output: one line per adapter with an FNV hash and the count of differing words)
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iterator>
using Microsoft::WRL::ComPtr;

static const UINT U0_BYTES = 4u << 20, U1_BYTES = 8u << 20;

static std::vector<uint8_t> Run(bool warp, const std::vector<char>& dxil, UINT groups, uint32_t seed, std::vector<uint8_t>& u0out, bool& ok, char* name) {
    ok = false;
    ComPtr<IDXGIFactory4> fac;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&fac));
    if (FAILED(hr)) { fprintf(stderr, "DXGI factory creation failed 0x%08x\n", (unsigned)hr); return {}; }
    ComPtr<IDXGIAdapter> ad;
    hr = warp ? fac->EnumWarpAdapter(IID_PPV_ARGS(&ad)) : fac->EnumAdapters(0, &ad);
    if (FAILED(hr)) { fprintf(stderr, "%s adapter enumeration failed 0x%08x\n", warp ? "WARP" : "GPU", (unsigned)hr); return {}; }
    DXGI_ADAPTER_DESC d; ad->GetDesc(&d); WideCharToMultiByte(CP_ACP, 0, d.Description, -1, name, 128, 0, 0);
    fprintf(stderr, "adapter: %s\n", name);
    ComPtr<ID3D12Device> dev;
    hr = D3D12CreateDevice(ad.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    if (FAILED(hr)) { fprintf(stderr, "D3D12 device creation failed 0x%08x\n", (unsigned)hr); return {}; }

    // raw UAVs through a descriptor table: unlike root UAVs they are bounds-checked, which matters because the shader follows pointers read from the data
    D3D12_DESCRIPTOR_RANGE range = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 0};
    D3D12_ROOT_PARAMETER rp[2] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;  rp[0].Descriptor = {0, 0};
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[1].DescriptorTable = {1, &range};
    const char* tablePath = getenv("DT_TABLE");
    D3D12_ROOT_SIGNATURE_DESC rs = {2, rp, 0, nullptr, tablePath ? D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED : D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> sb, eb;
    if (tablePath) {
        D3D12_DESCRIPTOR_RANGE1 range1 = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, 0};
        D3D12_ROOT_PARAMETER1 rp1[2] = {};
        rp1[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp1[0].Descriptor = {0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
        rp1[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp1[1].DescriptorTable = {1, &range1};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC vrs = {}; vrs.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        vrs.Desc_1_1 = {2, rp1, 0, nullptr, rs.Flags};
        D3D12SerializeVersionedRootSignature(&vrs, &sb, &eb);
    } else {
        D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sb, &eb);
    }
    ComPtr<ID3D12RootSignature> sig; dev->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&sig));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = sig.Get(); pd.CS = {dxil.data(), dxil.size()};
    ComPtr<ID3D12PipelineState> pso;
    hr = dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) { printf("  pipeline creation failed 0x%08x\n", (unsigned)hr); return {}; }

    auto mk = [&](UINT bytes, D3D12_HEAP_TYPE ht, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
        D3D12_HEAP_PROPERTIES hp = {ht}; D3D12_RESOURCE_DESC rd = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, fl};
        ComPtr<ID3D12Resource> r; dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r)); return r;
    };
    auto u0 = mk(U0_BYTES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    auto u1 = mk(U1_BYTES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    auto cb = mk(256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto up = mk(U0_BYTES, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto rb0 = mk(U0_BYTES, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto rb1 = mk(U1_BYTES, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    // input: words whose high byte is a small count (0..4) and whose low 24 bits are a random mask, as the shader reads them
    { uint32_t* p; up->Map(0, nullptr, (void**)&p); uint32_t s = seed;
      for (UINT i = 0; i < U0_BYTES / 4; i++) { s = s * 1664525u + 1013904223u; uint32_t cnt = (s >> 29) % 5; p[i] = (cnt << 24) | ((s >> 3) & 0xFFFFFF); if (getenv("DT_RAW")) { uint32_t x = s * 2654435761u; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; p[i] = x; } }
      up->Unmap(0, nullptr); }
    { uint32_t* c; cb->Map(0, nullptr, (void**)&c); c[0] = 0; c[1] = 0; c[2] = 0; c[3] = 0; cb->Unmap(0, nullptr); }

    std::vector<char> table;
    if (tablePath) { std::ifstream tf(tablePath, std::ios::binary); table.assign(std::istreambuf_iterator<char>(tf), {}); table.resize((table.size() + 255) & ~size_t(255)); }
    ComPtr<ID3D12Resource> tb, tup;
    if (tablePath) {
        tb = mk((UINT)table.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        tup = mk((UINT)table.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* p; tup->Map(0, nullptr, &p); memcpy(p, table.data(), table.size()); tup->Unmap(0, nullptr);
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, tablePath ? 3u : 2u, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> heap; dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
    UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    for (int i = 0; i < 2; i++) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {}; uv.Format = DXGI_FORMAT_R32_TYPELESS; uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uv.Buffer.NumElements = (i ? U1_BYTES : U0_BYTES) / 4; uv.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += (SIZE_T)i * inc;
        dev->CreateUnorderedAccessView(i ? u1.Get() : u0.Get(), nullptr, &uv, h);
    }
    if (tablePath) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {}; uv.Format = DXGI_FORMAT_R32_TYPELESS; uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uv.Buffer.NumElements = (UINT)table.size() / 4; uv.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += 2 * (SIZE_T)inc;
        dev->CreateUnorderedAccessView(tb.Get(), nullptr, &uv, h);
    }
    ComPtr<ID3D12CommandAllocator> al; dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al));
    ComPtr<ID3D12GraphicsCommandList> cl; dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al.Get(), nullptr, IID_PPV_ARGS(&cl));
    cl->CopyResource(u0.Get(), up.Get());
    if (tablePath) {
        cl->CopyResource(tb.Get(), tup.Get());
        D3D12_RESOURCE_BARRIER tbb = {}; tbb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; tbb.Transition.pResource = tb.Get();
        tbb.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; tbb.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; tbb.Transition.Subresource = 0xffffffff;
        cl->ResourceBarrier(1, &tbb);
    }
    D3D12_RESOURCE_BARRIER b[2] = {};
    for (int i = 0; i < 2; i++) { b[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b[i].Transition.pResource = i ? u1.Get() : u0.Get(); b[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b[i].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b[i].Transition.Subresource = 0xffffffff; }
    cl->ResourceBarrier(2, b);
    cl->SetComputeRootSignature(sig.Get()); cl->SetPipelineState(pso.Get());
    cl->SetComputeRootConstantBufferView(0, cb->GetGPUVirtualAddress());
    ID3D12DescriptorHeap* hs[] = {heap.Get()}; cl->SetDescriptorHeaps(1, hs);
    cl->SetComputeRootDescriptorTable(1, heap->GetGPUDescriptorHandleForHeapStart());
    cl->Dispatch(groups, 1, 1);
    for (int i = 0; i < 2; i++) { b[i].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE; }
    cl->ResourceBarrier(2, b);
    cl->CopyResource(rb0.Get(), u0.Get()); cl->CopyResource(rb1.Get(), u1.Get());
    cl->Close();
    ComPtr<ID3D12CommandQueue> q; D3D12_COMMAND_QUEUE_DESC qd = {}; dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
    ID3D12CommandList* lists[] = {cl.Get()}; q->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> f; dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f)); q->Signal(f.Get(), 1);
    HANDLE ev = CreateEvent(0, 0, 0, 0); f->SetEventOnCompletion(1, ev);
    if (WaitForSingleObject(ev, 60000) != WAIT_OBJECT_0) { printf("  dispatch timed out\n"); return {}; }
    if (FAILED(dev->GetDeviceRemovedReason())) { printf("  device removed 0x%08x\n", (unsigned)dev->GetDeviceRemovedReason()); return {}; }
    std::vector<uint8_t> out(U1_BYTES), o0(U0_BYTES);
    void* p; rb1->Map(0, nullptr, &p); memcpy(out.data(), p, U1_BYTES); rb1->Unmap(0, nullptr);
    rb0->Map(0, nullptr, &p); memcpy(o0.data(), p, U0_BYTES); rb0->Unmap(0, nullptr);
    if (const char* dp = getenv("DT_DUMP")) {            // DT_DUMP=<prefix>: write <prefix>_gpu|warp_u0.bin / _u1.bin
        for (int i = 0; i < 2; i++) { char fn[MAX_PATH]; sprintf_s(fn, "%s_%s_u%d.bin", dp, warp ? "warp" : "gpu", i); FILE* f = nullptr;
            if (!fopen_s(&f, fn, "wb")) { fwrite(i ? out.data() : o0.data(), 1, i ? out.size() : o0.size(), f); fclose(f); } }
    }
    u0out = o0; ok = true; return out;
}

static uint64_t Fnv(const std::vector<uint8_t>& v) { uint64_t h = 1469598103934665603ull; for (uint8_t c : v) { h ^= c; h *= 1099511628211ull; } return h; }

int main(int argc, char** argv) {
    setvbuf(stdout, 0, _IONBF, 0);
    if (argc < 2) { printf("usage: dispatch_test shader.dxil [groups] [seed]\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open shader: %s\n", argv[1]); return 2; }
    std::vector<char> dxil((std::istreambuf_iterator<char>(f)), {});
    if (dxil.empty()) { fprintf(stderr, "empty shader: %s\n", argv[1]); return 2; }
    UINT groups = argc > 2 ? atoi(argv[2]) : 4; uint32_t seed = argc > 3 ? atoi(argv[3]) : 1;
    std::vector<uint8_t> r[2], r0[2]; bool ok[2]; char name[2][128] = {};
    for (int w = 0; w < 2; w++) {
        r[w] = Run(w == 1, dxil, groups, seed, r0[w], ok[w], name[w]);
        if (ok[w]) { size_t nz = 0; for (size_t i = 0; i < r[w].size(); i += 4) if (*(uint32_t*)&r[w][i]) nz++;
            printf("%-34s output hash %016llx, %zu non-zero words; compressedBVH after hash %016llx\n", name[w], (unsigned long long)Fnv(r[w]), nz, (unsigned long long)Fnv(r0[w])); }
    }
    if (ok[0] && ok[1]) {
        size_t diff = 0, first = (size_t)-1;
        for (size_t i = 0; i < r[0].size(); i += 4) if (*(uint32_t*)&r[0][i] != *(uint32_t*)&r[1][i]) { if (first == (size_t)-1) first = i; diff++; }
        printf("%zu of %zu output words differ between the adapters%s", diff, r[0].size() / 4, diff ? "" : "\n");
        if (diff) printf(" (first at byte %zu: %08x vs %08x)\n", first, *(uint32_t*)&r[0][first], *(uint32_t*)&r[1][first]);
    }
    return ok[0] && ok[1] ? 0 : 1;
}
