"""
Dopamine Entropy Hardware Experiment
Simulates a Dopaminergic Controller (Tonic Search vs. Phasic Power-Gating) natively on MaxFormer.
"""
import sys, os, warnings, json
import torch
import torch.nn as nn
import numpy as np

# Use the imagenet subfolder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagenet"))
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

OUT_JSON = os.path.join(os.path.dirname(__file__), "analysis_outputs", "JSONs", "dopamine_experiment_results.json")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "10-384-T4.pth.tar")

NUM_HEADS = 6
IMG_SIZE = 224

class SpikeRecorder:
    def __init__(self):
        self.records = []
        self._handle = None

    def attach(self, module):
        self._handle = module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        self.records.append(output.detach().cpu().float())

    def detach(self):
        if self._handle:
             self._handle.remove()
             self._handle = None
    
    def clear(self):
        self.records.clear()

    @property
    def stacked(self):
        if not self.records: return None
        return torch.stack(self.records, dim=0)

def load_model():
    print("="*80)
    print("  [INIT] Loading MaxFormer for Dopaminergic Hardware Simulation")
    
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    import spikingjelly.clock_driven.neuron as _sj_neuron
    _orig_init = _sj_neuron.MultiStepLIFNode.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs["backend"] = "torch"
        _orig_init(self, *args, **kwargs)
    _sj_neuron.MultiStepLIFNode.__init__ = _patched_init

    from max_former import Max_Former
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Max_Former(in_channels=3, num_classes=1000, embed_dims=384, mlp_ratios=4, depths=10, T=4)

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    print(f"  [OK] Model successfully bridged to CPU/GPU on {device.upper()}")
    return model, device

def calculate_entropy(spike_tensor, num_heads):
    # spike_tensor shape: (T, B, C, N)  — where C = NUM_HEADS * D_HEAD
    T, B, C, N = spike_tensor.shape
    H = num_heads
    D = C // H
    # sum over T and D: shape (B, H, N)
    spk_h = spike_tensor.view(T, B, H, D, N).sum(dim=(0, 3))
    
    entropies = []
    total_spikes = 0
    # Process batch index 0
    for h in range(H):
        spatial_profile = spk_h[0, h, :].numpy()
        tot = spatial_profile.sum()
        total_spikes += tot
        if tot < 1e-4:
            entropies.append(0.0)
            continue
        p = spatial_profile / tot
        p = p[p > 0]
        # Spatial Entropy
        ent = -np.sum(p * np.log2(p + 1e-9))
        entropies.append(ent)
        
    return entropies, total_spikes

def main():
    model, device = load_model()
    
    # Identify SSA layers
    ssa_modules = []
    for name, module in model.named_modules():
        if "stage3" in name and "attn" in name and hasattr(module, "attn_lif"):
            ssa_modules.append((name, module.attn_lif))
            
    if not ssa_modules:
         print("Error: Could not locate SSA attn_lif blocks.")
         return

    print("="*80)
    print("  [EXPERIMENT] Executing Two-Pass Dynamic Dopaminergic Evaluations")
    
    results = {}
    N_SAMPLES = 10
    ENTROPY_THRESHOLD = 6.1  # Set to median to capture both Phase transitions

    for sample_id in range(1, N_SAMPLES + 1):
        x_sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        
        # ── PASS 1: Baseline Readout ──
        recorders = {name: SpikeRecorder().attach(lif) for name, lif in ssa_modules}
        with torch.no_grad():
            model(x_sample)
            
        block_entropies = []
        baseline_model_spikes = 0
        
        # Gather baseline states
        for name, lif in ssa_modules:
            rec = recorders[name]
            if rec.stacked is not None:
                spk = rec.stacked[0]
                ents, tot_spikes = calculate_entropy(spk, NUM_HEADS)
                block_entropies.append(np.mean(ents))
                baseline_model_spikes += tot_spikes
            rec.detach()
            rec.clear()
            
        global_entropy = float(np.mean(block_entropies))
        phase = "TONIC_SEARCH" if global_entropy > ENTROPY_THRESHOLD else "PHASIC_LOCK"
        
        # ── PASS 2: DOPAMINERGIC ACTUATION ──
        hooks = []
        
        if phase == "TONIC_SEARCH":
            # Globally lower V_th for all SSA lifs natively
            for name, lif in ssa_modules:
                lif.v_threshold = 0.5  # Massive excitability 
        else: # PHASIC_LOCK
            # Hard power-gate 4 out of 6 heads (the redundant 66%)
            def make_power_gate_hook(essential_heads=[0,1]):
                def hook(mdl, inputs, output):
                    T, B, C, N = output.shape
                    H = NUM_HEADS; D = C // H
                    out_v = output.clone().view(T, B, H, D, N)
                    for h in range(H):
                        if h not in essential_heads:
                            out_v[:, :, h, :, :] = 0.0 # Force physical sleep
                    return out_v.view(T, B, C, N)
                return hook

            for name, lif in ssa_modules:
                lif.v_threshold = 1.0 # Guarantee reset
                # Just lock onto Head 0 and 1, completely power-down 2,3,4,5
                hdl = lif.register_forward_hook(make_power_gate_hook(essential_heads=[0,1]))
                hooks.append(hdl)
                
        # Re-attach recorders for readout
        recorders = {name: SpikeRecorder().attach(lif) for name, lif in ssa_modules}
        with torch.no_grad():
             model(x_sample)
             
        actuated_model_spikes = 0
        for name, lif in ssa_modules:
            rec = recorders[name]
            if rec.stacked is not None:
                spk = rec.stacked[0]
                _, tot_spikes = calculate_entropy(spk, NUM_HEADS)
                actuated_model_spikes += tot_spikes
            rec.detach()
            rec.clear()
            
        # Clean up hooks and restore states
        for h in hooks: h.remove()
        for name, lif in ssa_modules: lif.v_threshold = 1.0 # Restore baseline
        
        delta_spikes = actuated_model_spikes - baseline_model_spikes
        delta_pct = (delta_spikes / max(1, baseline_model_spikes)) * 100.0
        
        print(f"Sample {sample_id:02d} | Entropy: {global_entropy:.3f} | Phase: {phase:<12} | Baseline MACs: {baseline_model_spikes:<7.0f} | Final MACs: {actuated_model_spikes:<7.0f} | Shift: {delta_pct:+.1f}%")

        results[f"Sample_{sample_id:02d}"] = {
            "global_entropy": float(global_entropy),
            "dopamine_phase": phase,
            "baseline_spike_operations": float(baseline_model_spikes),
            "actuated_spike_operations": float(actuated_model_spikes),
            "dynamic_power_shift_pct": float(delta_pct)
        }
        
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
         json.dump(results, f, indent=4)
         
    print("="*80)
    print(f"  [SUCCESS] Results structured directly to {OUT_JSON}")
    print("="*80)

if __name__ == "__main__":
     main()
