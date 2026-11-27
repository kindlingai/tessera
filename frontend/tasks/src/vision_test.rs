use super::api::{EmbedMessage, EmbedPart, Image};
use super::png;
use super::vision::{
    IMAGE_TOKEN, ImageSpan, MAX_PIXELS, MIN_PIXELS, decode_base64, decode_image, decode_uri,
    encode_image, expand_images, format_chat, mrope_ids, pack_text, pack_vision, pos_bilinear,
    resize_bicubic, smart_resize, synthetic_rgb,
};

fn b64_enc(bytes: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for g in bytes.chunks(3) {
        let n = (u32::from(g[0]) << 16)
            | (u32::from(*g.get(1).unwrap_or(&0)) << 8)
            | u32::from(*g.get(2).unwrap_or(&0));
        for i in 0..4 {
            out.push(if i > g.len() {
                '='
            } else {
                ALPHA[((n >> (18 - 6 * i)) & 63) as usize] as char
            });
        }
    }
    out
}

#[test]
fn resize_and_decode() {
    assert_eq!(
        smart_resize(512, 512, MIN_PIXELS, MAX_PIXELS).unwrap(),
        (512, 512)
    );
    assert_eq!(
        smart_resize(16, 16, MIN_PIXELS, MAX_PIXELS).unwrap(),
        (64, 64)
    );
    let (h, w) = smart_resize(4000, 3000, MIN_PIXELS, MAX_PIXELS).unwrap();
    assert!(h * w <= MAX_PIXELS && h % 32 == 0 && w % 32 == 0);
    assert!(smart_resize(0, 64, MIN_PIXELS, MAX_PIXELS).is_err());
    assert!(smart_resize(10000, 10, MIN_PIXELS, MAX_PIXELS).is_err());
    let solid = vec![64u8; 64 * 64 * 3];
    let resized = resize_bicubic(&solid, 64, 64, 32, 32);
    assert_eq!(resized.len(), 32 * 32 * 3);
    assert!(resized.iter().all(|&b| b == 64));
    let rgba = Image {
        width: 32,
        height: 32,
        rgba: vec![128u8; 32 * 32 * 4],
    };
    let png_bytes = png::encode(&rgba);
    let uri = format!("data:image/png;base64,{}", b64_enc(&png_bytes));
    let raw = decode_uri(&uri).unwrap();
    assert_eq!(raw, png_bytes);
    assert_eq!(decode_base64(&b64_enc(b"hello")).unwrap(), b"hello");
    assert_eq!(decode_base64(" aGVs\n bG8= ").unwrap(), b"hello");
    assert!(decode_base64("ab=c").is_err());
    assert!(decode_base64("abc").is_err());
    let big: Vec<u8> = (0..300_000usize).map(|i| (i * 31 + 7) as u8).collect();
    assert_eq!(decode_base64(&b64_enc(&big)).unwrap(), big);
    let rgb = decode_image(&raw).unwrap();
    assert_eq!((rgb.width, rgb.height), (32, 32));
    let prep = encode_image(&rgb, MIN_PIXELS, MAX_PIXELS, 48).unwrap();
    assert_eq!((prep.vit_h, prep.vit_w, prep.tokens()), (4, 4, 4));
    // Parallel large resize and patchify match expected dimensions and range.
    let grad = synthetic_rgb(640, 480, 3);
    let big_enc = encode_image(&grad, MIN_PIXELS, MAX_PIXELS, 48).unwrap();
    assert_eq!((big_enc.vit_h, big_enc.vit_w), (30, 40));
    let vis_big = pack_vision(&[&big_enc]);
    assert_eq!(vis_big[0].shape, [300, 4, 1536]);
    assert!(
        vis_big[0]
            .floats()
            .unwrap()
            .iter()
            .all(|&v| (-1.0..=1.0).contains(&v))
    );
}

#[test]
fn pack_vision_and_text() {
    let im = synthetic_rgb(64, 64, 1);
    let prep = encode_image(&im, MIN_PIXELS, MAX_PIXELS, 48).unwrap();
    let (pos_ids, pos_w) = pos_bilinear(48, 4, 4);
    assert_eq!((pos_ids.len(), pos_w.len()), (16 * 4, 16 * 4));
    assert!((pos_w[0] + pos_w[1] + pos_w[2] + pos_w[3] - 1.0).abs() < 1e-6);
    let vis = pack_vision(&[&prep]);
    assert_eq!(vis.len(), 5);
    assert_eq!(vis[0].shape, [4, 4, 1536]);
    assert_eq!(vis[1].shape, [4, 4, 4]);
    assert_eq!(vis[3].shape, [4, 4, 2]);
    let spans = [ImageSpan {
        pos: 2,
        vit_h: 4,
        vit_w: 4,
    }];
    let ids = mrope_ids(8, &spans);
    assert_eq!(
        (ids[0], ids[2], ids[5], ids[6]),
        ([0, 0, 0], [2, 2, 2], [2, 3, 3], [4, 4, 4])
    );
    let msgs = vec![EmbedMessage {
        role: "user".into(),
        parts: vec![EmbedPart::Image(vec![1]), EmbedPart::Text("hi".into())],
    }];
    let (prompt, imgs) = format_chat(&msgs, true);
    assert_eq!(imgs.len(), 1);
    assert!(prompt.contains("<|vision_start|><|image_pad|><|vision_end|>hi"));
    let (expanded, got_spans) =
        expand_images(&[10, IMAGE_TOKEN, 20], IMAGE_TOKEN, &[prep]).unwrap();
    assert_eq!(
        expanded,
        vec![10, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 20]
    );
    let packed = pack_text(&[expanded.as_slice()], &[got_spans.as_slice()]);
    assert_eq!(packed.len(), 5);
    assert_eq!(packed[0].shape, [6]);
    assert_eq!(packed[1].shape, [6, 3]);
    assert_eq!(packed[3].shape, [1]);
}
