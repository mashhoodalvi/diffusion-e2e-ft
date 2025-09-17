from diffusers import DiffusionPipeline
import diffusers
import os

os.makedirs("outputs", exist_ok=True)

image = diffusers.utils.load_image(
    "https://gonzalomartingarcia.github.io/diffusion-e2e-ft/static/lego.jpg"
)

# Depth
pipe = DiffusionPipeline.from_pretrained(
    "GonzaloMG/marigold-e2e-ft-depth",
    custom_pipeline="GonzaloMG/marigold-e2e-ft-depth",
).to("cuda")
depth = pipe(image)
pipe.image_processor.visualize_depth(depth.prediction)[0].save("outputs/depth.png")
pipe.image_processor.export_depth_to_16bit_png(depth.prediction)[0].save("outputs/depth_16bit.png")


# Normals
pipe = DiffusionPipeline.from_pretrained(
    "GonzaloMG/stable-diffusion-e2e-ft-normals",
    custom_pipeline="GonzaloMG/marigold-e2e-ft-normals",
).to("cuda")
normals = pipe(image)
pipe.image_processor.visualize_normals(normals.prediction)[0].save("outputs/normals.png")