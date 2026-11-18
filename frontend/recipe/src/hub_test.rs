use crate::hub::{safe_path, wanted};

fn owned(list: &[&str]) -> Vec<String> {
    list.iter().map(|n| n.to_string()).collect()
}

#[test]
fn diffusers_folders() {
    let keep = [
        "model_index.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "tokenizer/merges.txt",
        "transformer/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ];
    let mut names = owned(&keep);
    let drop = [
        "flux-2-klein-4b.safetensors",
        "vae/x.bin",
        "assets/a.png",
        "extra/a.json",
    ];
    names.extend(owned(&drop));
    let folders = owned(&["text_encoder", "tokenizer", "transformer", "vae"]);
    assert_eq!(wanted(&names, Some(&folders)), keep);
}

#[test]
fn plain_repo() {
    let names = owned(&[
        "config.json",
        "model.safetensors",
        "x.bin",
        "onnx/config.json",
    ]);
    assert_eq!(wanted(&names, None), ["config.json", "model.safetensors"]);
    assert!(safe_path("vae/config.json"));
    for bad in ["../x.json", "a/../b.json", "/etc/x.json", "a\\b.json"] {
        assert!(!safe_path(bad), "{bad}");
    }
}
