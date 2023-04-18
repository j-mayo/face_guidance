from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, DPMSolverMultistepScheduler, KDPM2AncestralDiscreteScheduler
from lora_diffusion import monkeypatch_lora, tune_lora_scale, patch_pipe
import torch
import torch.nn.functional as F
import clip
from PIL import Image
import os
from kornia.geometry.transform import warp_affine

import cv2
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
# for face detection
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from diffusers.utils import randn_tensor

from arcface_torch.backbones import get_model

from arcface_torch.backbones import get_model

#from insightface.recognition.arcface_mxnet.common.build_eval_pack import get_norm_crop
from build_eval_pack import get_norm_crop, get_norm_crop2

import argparse

parser = argparse.ArgumentParser(description='Inference lora with iterative face identity injection')
parser.add_argument('--lora_path', type=str, default=None, help='path of lora weight which you want to use')
parser.add_argument('--pretrained_model_name', type=str, default="stabilityai/stable-diffusion-2-1-base", help='name of model card at huggingface which you want to use')
parser.add_argument('--dataset_path', type=str, default="dataset/cropped", help='path of dataset at used to learn lora weight')
parser.add_argument('--N', type=int, default=1, help='number of iterative whole diffusion sampling')
parser.add_argument('--skip', type=int, default=25, help='number of skip timesteps, in denoising processing')
parser.add_argument('--step', type=int, default=50, help='number of denoising processing step')
parser.add_argument('--output_folder_prefix', type=str, default="", help='the prefix of name of output folder')
parser.add_argument('--start_scheduler', type=str, default="euler", help='choose among of euler, dpm, kdpm')

args = parser.parse_args()

# model, preprocess = clip.load("ViT-B/32", device="cuda")
loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

"""
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True,return_tensors="pt")
text_features = model.get_text_features(**inputs)
print(text_features.shape)
"""


def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask

def get_face_feature_from_tensor(img_shape, img, dtype='uint8', app=None):
    # 이미지 tensor로부터 face feature 추출
    # use arcface, return the masked face bbox
    # img = img[0]
    # img is in [0, 1]
    #img = img.permute(0, 2, 3, 1)
    detection_img = img[0].clone().detach()
    detection_img = detection_img.to(dtype=torch.float32)
    detection_img = detection_img.cpu().numpy()
    detection_img = (detection_img * 255).round()
    prev_dtype = img.dtype
    #print(img.shape)
    
    detection_img = cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_tensor.jpg", detection_img)
    height, width = img.shape[1:3]
    mask = np.zeros((height, width, 1), dtype=dtype)

    img = img.permute(0, 3, 1, 2)
    #img.sub_(0.5).div_(0.5)
    wraped_output = get_norm_crop2(detection_img, img)
    if wraped_output is not None:
        warped_img, det = wraped_output
        det = [int(x) for x in det[0]]
        # 사이즈가 112만 가능한듯?
        warped_img.sub_(0.5).div_(0.5)
        face_feature = net(warped_img)
        #img.div_(2).sub(-0.5)
        #warped_img = warped_img.to(dtype=prev_dtype)

        mask[det[1]:det[3], det[0]:det[2]] = 1
        mask = torch.from_numpy(mask).cuda().permute(2, 0, 1).unsqueeze(0)
        box = det

        print("***face is detected!***")
    else: # if there is no face, return None mask
        #h, w = detection_img.shape[:2]
        #mask = bbox2mask(detection_img.shape[1:], (h//4, w//4, h//2, w//2))
        mask = None
        face_feature = None
        box = None
        print("***face is not detected!***")

    return mask, face_feature, box


def get_face_feature_from_folder(path, h, w, device):
    # path 경로의 폴더로부터 embedding 추출해서 평균과 embedding들의 list를 반환
    # 기존 데이터
    filelist = os.listdir(path)

    #app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    #app.prepare(ctx_id=0, det_size=(h, w))

    os.makedirs(path+"_face", exist_ok=True)
    num_data = 0
    features = torch.zeros((1, 512)).to(device=device)
    feature_list = []
    for name in filelist:
        num_data += 1
        img = cv2.imread(path+"/"+name)
        img_aligned, det = get_norm_crop(path+"/"+name)

        det = [int(x) for x in det[0]]
        cv2.imwrite("test.jpg", img[det[1]:det[3], det[0]:det[2]])
        img = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(device=device)
        img.div_(255).sub_(0.5).div_(0.5)
        #img = F.interpolate(img, size=(112, 112), mode=Mode)
        with torch.no_grad():
            faces = net(img)
            feature_list.append(faces[0])
        features += faces

    return features / num_data, feature_list

def step(pipe,
         app,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        condition: Optional[List[torch.FloatTensor]] = None,
        return_dict: bool = True,
        encoder_hidden_state: torch.FloatTensor = None,
        mask: Optional[torch.FloatTensor] = None,
        lam: Optional[float] = 1.0,
        do_classifier_free_guidance: Optional[bool] = True,
        ):
        # This step function is the "modified version" of EulerAncestralDiscrete's step function.
        # For the face-guidance, I put some codes.


        target = torch.tensor([1]).to(device="cuda")
        pipe_scheduler = pipe.scheduler
        #print("timestep: ", timestep)
        with torch.enable_grad():
            temp_sample = sample.detach().requires_grad_()
            temp_sample_input = torch.cat([temp_sample] * 2) if do_classifier_free_guidance else temp_sample
            # Need to calculate gradient, so re-get the model output with enable grad
            temp_model_output = pipe.unet(temp_sample_input, timestep, encoder_hidden_states=encoder_hidden_state).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = temp_model_output.chunk(2)
                temp_model_output = noise_pred_uncond + 7 * (noise_pred_text - noise_pred_uncond)
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(pipe_scheduler.timesteps.device)

            step_index = (pipe_scheduler.timesteps == timestep).nonzero().item()
            sigma = pipe_scheduler.sigmas[step_index]

            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            if pipe_scheduler.config.prediction_type == "epsilon":
                pred_original_sample = temp_sample - sigma * temp_model_output
            elif pipe_scheduler.config.prediction_type == "v_prediction":
                # * c_out + input * c_skip
                pred_original_sample = temp_model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (temp_sample / (sigma**2 + 1))
            elif pipe_scheduler.config.prediction_type == "sample":
                raise NotImplementedError("prediction_type not implemented yet: sample")
            else:
                raise ValueError(
                    f"prediction_type given as {pipe_scheduler.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
                )

            sigma_from = pipe_scheduler.sigmas[step_index]
            sigma_to = pipe_scheduler.sigmas[step_index + 1]
            sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

            """
            # added: tweedie's formula
            alpha_prod_t = pipe_scheduler.alphas_cumprod[step_index]
            alpha_prod_t_prev = pipe_scheduler.alphas_cumprod[step_index+1]
            beta_prod_t = 1 - alpha_prod_t
            original_sampel_hat = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            """
            # using face embedding, calculate loss and grad for z_t latent
            pred_original_image = decode_latents(pipe, pred_original_sample)
            # denoising image

            new_mask, cur_embedding, _ = get_face_feature_from_tensor(pred_original_image.shape, pred_original_image, app=app)
            if new_mask is not None:
                new_mask = torch.nn.functional.interpolate(
                    new_mask, size=(new_mask.shape[2] // pipe.vae_scale_factor, new_mask.shape[3] // pipe.vae_scale_factor)
                ).to(dtype=model_output.dtype)
            if cur_embedding is not None and timestep > 75:
                # condition
                cur_loss = 0
                for emb in condition:
                    cur_loss += lam * loss(emb.unsqueeze(0), cur_embedding, target)
                #cur_loss = lam * loss(cur_embedding, condition, target)
                #print("loss: ", cur_loss)
                cur_loss /= len(cur_embedding)
                grad = torch.autograd.grad(cur_loss, temp_sample)[0]
                #_, grad = grad.chunk(2)

            else:
                grad = 0
            # 2. Convert to an ODE derivative
            derivative = (sample - pred_original_sample) / sigma

            dt = sigma_down - sigma

            prev_sample = sample + derivative * dt

            device = model_output.device
            noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
            
            prev_sample = prev_sample + noise * sigma_up

            # gradient, x_t-1 <- x_t-1 - dl/dx_t
            # lambda scaling? timestep 너무 마지막엔 loss 추가하지 않게?
            if timestep > 75:
                new_prev_sample = prev_sample - grad * new_mask if new_mask is not None else prev_sample - grad# 여기부터...
            else:
                new_prev_sample = prev_sample

            if not return_dict:
                return (new_prev_sample,)
            if cur_embedding is not None:
                cur_embedding = cur_embedding.clone().detach()
        return {"prev_sample":new_prev_sample, "pred_original_sample":pred_original_sample, "embedding": cur_embedding}


def decode_latents(pipe, latents, t=None, new=False):
        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.permute(0, 2, 3, 1).float()
        numpy_image = image.detach().cpu().numpy()
        numpy_image = pipe.numpy_to_pil(numpy_image)[0]
        numpy_image.save("dfsdfsad.jpg") # 관찰용 이미지
        return image


# refine 시의 코드
def inference_withfaceguide(
        pipe,
        start_image,
        N,
        skip_timestep = 25,
        condition_embedding: Optional[torch.FloatTensor] = None,
        condition_embedding_list: Optional[List[torch.FloatTensor]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        device="cuda",
        lam=0.5,
    ):

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(start_image.shape[1], start_image.shape[2]))
    dtype = start_image.dtype
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor

    pipe.check_inputs(
        prompt, height, width, callback_steps
    )
    unet = pipe.unet
    scheduler = pipe.scheduler
    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ).to(dtype=dtype, device=device)

    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    # timesteps = timesteps[skip_timestep:] # skip some timesteps

    num_channels_latents = pipe.unet.in_channels

    # prepare image latent
    noise = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    ).to(dtype=dtype, device=device)
    noise /= pipe.scheduler.init_noise_sigma

    # get face mask
    mask, cur_embedding, bbox = get_face_feature_from_tensor(start_image.shape, start_image/2+0.5, app=app)
    if mask == None:
        return None, None, None
    mask = mask.to(device=device)

    start_image = start_image.permute(0, 3, 1, 2)

    # cur_embedding = cur_embedding.to(device=device)
    mask = torch.nn.functional.interpolate(
        mask, size=(height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    ).to(dtype=noise.dtype)
    start_image = start_image.cuda().to(dtype=noise.dtype, device=device)
    latents = pipe.vae.encode(start_image).latent_dist.sample(None) * 0.18215 # not noised latent * vae scaling factor
    origin_latents = latents.clone().detach()
    latent_timestep = timesteps[skip_timestep * pipe.scheduler.order:skip_timestep * pipe.scheduler.order+1].repeat(batch_size * num_images_per_prompt)
    non_facial_gt = latents * (1 - mask)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    with torch.no_grad():
        with pipe.progress_bar(total=(num_inference_steps-skip_timestep)*N) as progress_bar:
            for z in range(N):
                non_facial_latents = latents * (1 - mask)
                latents = scheduler.add_noise(latents, noise, latent_timestep)
                noised_non_facial_gt = scheduler.add_noise(non_facial_gt, noise, latent_timestep) * (1 - mask)
                latents = noised_non_facial_gt + latents * mask

                num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
                
                for i, t in enumerate(timesteps):

                    if i < skip_timestep:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    # non_facial_latents = latents * (1 - mask)
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    # face guidance 부분
                    step_output = step(pipe, app, noise_pred, t, latents, condition=condition_embedding_list, encoder_hidden_state=prompt_embeds, mask=mask, lam=lam, do_classifier_free_guidance=do_classifier_free_guidance, **extra_step_kwargs)
                    latents = step_output["prev_sample"]
                    
                    # for face guidance
                    # 준하님이 지적하셨던 기존 latent에 noise 끼얹은 부분
                    non_facial_noise = scheduler.add_noise(origin_latents, noise, torch.tensor(t).repeat(batch_size * num_images_per_prompt))
                    latents = latents * mask + non_facial_noise * (1 - mask)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

    embedding = step_output["embedding"]
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()

    # 9. Run safety checker
    if hasattr(pipe, 'safety_cheker'):
        image, has_nsfw_concept = pipe.run_safety_checker(image, device, prompt_embeds.dtype)

    # 10. Convert to PIL
    image = pipe.numpy_to_pil(image)
    return image, bbox, embedding

    
# settings

schedulers = {"euler": EulerAncestralDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
    "kdpm": KDPM2AncestralDiscreteScheduler,
}

# face embedding을 계산할 기존 dataset path
source_path = args.dataset_path
# arcface model path
face_model_path = "arcface_torch/glint360k_cosface_r100_fp16_0.1/backbone.pth"

# for face align
net = get_model('r100', fp16=True)
net.load_state_dict(torch.load(face_model_path))
net.to("cuda")
net.eval()

# model id name (huggingface)
model_id ="stabilityai/stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name, torch_dtype=torch.float16).to(
    "cuda"
)
#pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = schedulers[args.start_scheduler].from_config(pipe.scheduler.config)

# 저 부분에 원하는 lora tensor 경로를 주면 합쳐줍니다
#"output/lora-sd2_1-512-notfaceseg/final_lora.safetensors",
#"output/mixed-512-3-5.safetensors"
patch_pipe(
    pipe,
    args.lora_path,
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)

# example prompts... 귀찮아서 몇 개 만들어놓고 돌려 썼어요
"""
example_prompts = [
    "portrait of <minji> by mario testino 1950, 1950s style, hair tied in a bun, taken in 1950, detailed face of <minji>, sony a7r",
    "photo of <minji>, 50mm, sharp, muscular, detailed realistic face, hyper realistic, perfect face, intricate, natural light, <minji> underwater photoshoot,collarbones, skin indentation, Alphonse Mucha, Greg Rutkowski",
    "a photo of <minji> in advanced organic armor, biological filigree, detailed symmetric face, flowing hair, neon details, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, octane, art by Krenz Cushart , Artem Demura, Alphonse Mucha, digital cgi art 8K HDR by Yuanyuan Wang photorealistic",
    "a photo of <minji> on the beach, small waves, detailed symmetric face, beautiful composition",
    "a photo of <minji> rainbow background, wlop, dan mumford, artgerm, liam brazier, peter mohrbacher, jia zhangke, 8 k, raw, featured in artstation, octane render, cinematic, elegant, intricate, 8 k",
    "photo of Summoner <minji> with a cute water elemental, fantasy illustration, detailed face, intricate, elegant, highly detailed, digital painting, artstation, concept art, wallpaper, smooth, sharp focus, illustration, art by artgerm and greg rutkowski",
    "<minji>, cyberpunk 2077, 4K, 3d render in unreal engine",
    "a pencil sketch of <minji>",
    "a photo of <minji>, realistic, best quality, masterpiece",
]

example_prompts = ["a photo of a <minji> woman with style <s1>, forest background, realistic face, detailed symmetric face, flowing hair",
                "a photo of <minji> with style <s1>",
                "a pencil sketch of <minji> with style <s1>",
                "photo of <minji> with style <s1>, 50mm, sharp, detailed realistic face, hyper realistic, perfect face, intricate, natural lightn",
                "a photo of a <minji>, with style <s1> and white background, realistic face"]

"""

example_prompts_mix = ["a photo of a <sks1>, with style <style1>, realistic face, flowing hair, natural skin texture, 24mm",
                "a photograph of <sks1>, with style <style1>, alone, 1girl",
                "a pencil sketch of <sks1> with style <style1>",
                "photo of <sks1> with style <style1>, 50mm, sharp, detailed realistic face, hyper realistic, perfect face, intricate, natural lightn",
                "a photo of a solo, centered <sks1>, with style <style1> and white background, realistic face, natural skin texture, 24mm",
                "<sks1> in a construction outfit with style <style1>"]

example_prompts2 = ["a photo of a <sks1><sks2> woman with style <style1>, forest background, realistic face, detailed symmetric face, flowing hair",
                "a photo of <sks1><sks2> with style <style1>",
                "a pencil sketch of <sks1><sks2> with style <style1>",
                "photo of <sks1><sks2> with style <style1>, 50mm, sharp, detailed realistic face, hyper realistic, perfect face, intricate, natural lightn",
                "a photo of a <sks1><sks2>, with style <style1> and white background, realistic face"
]



negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
negative_prompt2 = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans,extra fingers,fewer fingers,strange fingers,bad hand,"
negative_prompt3 = "(semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
negative_promptRV = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"


tune_lora_scale(pipe.unet, 1.00)

torch.manual_seed(42)

N = args.N
folder_name = "faceguide_images_tests/"+args.output_folder_prefix+"_"+str(N)+"_feature"
os.makedirs(folder_name, exist_ok=True)

ff = open("similarity.txt", "w")
target = torch.tensor(1).cuda()

for i, prompt in enumerate(example_prompts_mix):
    N = args.N
    # 이미지 먼저 생성
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7, output_type="img").images
    start_image = pipe.numpy_to_pil(image)[0]
    #image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
    start_image.save(folder_name+"/start_prompt_"+args.start_scheduler+str(i)+".jpg")
    print("start image is saved at "+folder_name+"/start_prompt_"+args.start_scheduler+str(i)+".jpg")

    image = torch.from_numpy(image).to(dtype=torch.float16, device="cuda")
    # this shape is N, H, W, C
    # image = image.permute(0, 3, 1, 2)
    image = 2.0 * image - 1.0
    img = image.clone().detach()
    face_feature, feature_list = get_face_feature_from_folder(source_path, 512, 512, "cuda")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    for lam in [0.1, 0.25, 0.5, 1, 5, 10, 15, 20, 25, 37]:
        ff.write(f"*****Image {i}, lambda {lam}, similarity between original dataset and generated image: *****\n\n")
        # refine image with face embedding
        image, bbox, final_embedding = inference_withfaceguide(
            pipe, 
            img, 
            N, 
            prompt = prompt, 
            condition_embedding=face_feature, 
            condition_embedding_list = feature_list, 
            num_inference_steps=args.step, 
            negative_prompt=negative_promptRV, 
            guidance_scale=7, 
            lam=lam, 
            skip_timestep=args.skip,
        )
        if image == None:
            print("Can't fine the face in start image... pass this prompt\n")
            continue
        image = image[0]
        image.save(folder_name+"/_prompt_"+str(i)+"_"+args.start_scheduler+"_negatived_lam_"+str(lam)+".jpg")

        for num, origin_embedding in enumerate(feature_list):
            if final_embedding is not None:
                sim = 1 - loss(origin_embedding, final_embedding[0], target)
                ff.write(f"Cosine similarity between original image {num} and generated image: {sim}, theta: {torch.rad2deg(torch.arccos(sim))}\n")
            else:
                ff.write(f"No... final face embedding for image {i}, lambda {lam} is None...\n")

ff.close()