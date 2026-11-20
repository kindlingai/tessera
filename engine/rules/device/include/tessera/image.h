// One denoising step over a batch of latents.

#ifndef TESSERA_CAP_IMAGE
#error "import <tessera/1/image>"
#endif

#ifndef TESSERA_IMAGE_H
#define TESSERA_IMAGE_H

namespace tessera {

struct image_step {
    u32 batch;
    u32 height;
    u32 width;
    u32 channels;
    ref timestep;
    ref guidance;
    ref seed;
};

#if !defined(__METAL_VERSION__)

TESSERA_FN const image_step* image(const table* t) {
    return uses<image_step>(t, IMAGE);
}

#endif

}  // namespace tessera

#endif  // TESSERA_IMAGE_H
