# face_guidance

사용법을 적은 md 파일입니다
어짜피 계정 문제는 없으니, 사용법만 서술하고자 합니다.

------------------


사용 폴더: sjy/lora


train script: 
train.sh, train_style.sh
이 script는 lora에서 나온 것과 거의 같으니, 인자 설정만 해서 훈련 돌리면 됩니다

--------------


inference script: 
inference.py (기본 lora), inference_faceguide.py (face identity injection을 추가한 lora, 미팅 때 설명드렸던 코드에요), inference_unclip.py (unclip 실험할때 잠시 만들었던 파일. 그런데 unclip 성능이 애매해서 일단 제쳐두었는데, 아예 증명사진 segmentation?을 condition 으로 주면 또 어떨지 모르겠네요)

이쪽은 제가 arg를 받는 식으로 코드를 짜지는 않았고, 각 파일의 위쪽에서 model_id, patch할 lora weight, prompt 등을 관리해준 뒤
밑의 for문에서 inference 시, 생성된 이미지를 저장할 폴더 부분만 신경쓰면 될 거에요
    
inference_faceguide.py의 경우엔 주석을 약간 달아서 그 부분 읽어보시면 될 것 같고, 역시 다른 inference 파일처럼 model_id, lora weight 등의 부분이 중간 정도에 있어요  

실행은 이렇게 합니다.
```shell
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE python inference_faceguide.py \
--lora_path (path of mixed lora weight) \
--pretrained_model_name (name of model card of huggingface) \
--dataset_path (dataset which is used for lora learning) \
--N (number of refine, with face guidance) \
--skip (number of skip timestep) \
--step (number of inference timestep) \
--output_folder_prefix (output 폴더 앞에 구별을 위해 붙이고 싶은 단어?) \
--start_scheduler (euler, dpm, kdpm 중 선택) 
```

ex)
```shell
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE python inference_faceguide.py \
--lora_path output_mixed/mixed-RV2-minji-7-3-f-nf.safetensors \
--pretrained_model_name SG161222/Realistic_Vision_V2.0 \
--dataset_path dataset/cropped \
--N 1 \
--skip 25 \
--step 50 \
--output_folder_prefix RV2-minji-7-3-f-nf \
--start_scheduler euler 
```

(faceguide.sh에 제가 실험하느라 몇 개 만들어놓아서, 그거 복붙해서 적절히 쓰셔도 됩니다)


lora 학습 결과들이 있는 폴더: lora/output
이전데 기본으로 사용했던 건 lora_sd2_1_512-notfaceseg
lora_sd2_1_base_512-style (style)
이였고, 이후 mixed version은 mixed-512-3-5.safetensor 사용하고 있었어요
테스트 중인 게 좀 많아서 폴더가 난잡합니다 ... 2개 단어로 하는 것도 시도해보고 했는데 좋지는 않더라고요

(mixed-x-y는 lora와 스타일 lora를 0.x, 0.y 만큼 합쳤다는 뜻입니다)

폴더명으로 어느정도 알 수 있는데, sd2-1: SD 2.1버전 사용, 1_5는 1.5버전 사용
minji: 민지 데이터셋으로, anya: 퀸스갬빗 주인공 데이터셋으로
뒤에 붙은 deliberate, RV2: 각각 civitai에서 발견했던 모델 사용해서 튜닝한 것 - 이 경우 sd2_1은 무시하셔도 됩니다. 관성적으로 폴더명이 저렇게 지어졌네요
faced: 얼굴만 crop된 사진으로 학습된 것

style의 경우, nonfaced가 붙어 있음 - 얼굴 마스킹된 사진으로 학습된 것!


dataset 폴더: lora/dataset

    cropped: 사이즈를 맞춰준 민지 사진
    minji: 사이즈가 각기 다른 상태의 민지 사진
    jeongjae: 이정재 사진
    Style_dataset: style 사진
    Style_nonfaced: face를 마스킹한 style 사진
    style_sihyun: 시현하다 사진 중 100장? 정도 모아놓은 사진들
    style_sihyun_nonfaced: face mask한 사진들
    anya: 퀸스갬빗 주인공 사진(5장)
    class_dataset_ffhq : prior preservation에 사용하는, ffhq에서 긁어온 200장의 사람 


similarity.txt : inference_faceguide.py 코드 실행 시, 민지 사진들과 최종 생성된 사진들의 similarity를 구해서 이 파일에 작성합니다.

align의 경우, inference_faceguide.py에서 진행되는데, 이건 lora 폴더 안에 insightface를 아예 가져와서 했고요
build_eval_pack.py라는 파일에서 import한 함수를 통해 align을 진행해요. 이 부분이 좀 복잡하게 얽혀 있긴 해서, 만약 이해가 안 가는 부분은 제게 말씀해주세요!
