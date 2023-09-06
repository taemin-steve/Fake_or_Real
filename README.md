# Fake_or_Real
<!--![thum (1)](https://github.com/taemin-steve/Fake_or_Real/assets/75752289/2c628a4f-d93b-4c44-9ac5-f6ec6ba5a7df)-->
<div style="width: 100%; float: center;">
    <img src="https://github.com/taemin-steve/Fake_or_Real/assets/75752289/2c628a4f-d93b-4c44-9ac5-f6ec6ba5a7df" alt="Image 1">
</div>

## _Introduction_
[AI CONNECT Fake or Real](https://aiconnect.kr/competition/detail/227/task/295/taskInfo) 생성이미지 판별 경진대회 9위 구현 코드 입니다. 
다양한 생성 모델로 인해 영상 위변조의 문제가 대두되는 상황 속에서 생성 AI가 만들어낸 가짜(Fake) 이미지와 진짜(Real) 이미지를 분류하는 문제입니다. 

## _Model_
다양한 모델을 사용해 보았으나, 최종적으로 Efficient Net을 사용하였습니다. 
Train data에 대하여 overfitting이 심하였고, 보다 복잡한 모델을 사용하는 것은 **overfitting 오히려 심하게 할것이라 판단**하였기에, 효율적으로 학습을 진행할 수 있는 Efficient Net을 기반으로 진행하였습니다. 

## _Data Augmentation_ 
Validation에서 accuracy, f1 score 모두 99%에 육박하지만, 제출시에 f1 score가 70로 감소하는 상황이였기에 Overfitting을 해결하는 것이 가장 중요하다고 판단, 가장 확실한 방법이 Data Augmentation을 다양한 방법으로 시행하였습니다. 

##### 1) 외부 데이터 활용 
본 대회는 특이하게도 외부 데이터가 허용되어있는 대회였기에 외부 데이터를 적극 활용하기로 하였습니다. 
Kaggle에서 [Dalle](https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset), [Midjourney](https://www.kaggle.com/datasets/ldmtwo/midjourney-250k-csv?select=midjourney_2022_reduced.csv), [Stable Diffusion](https://www.kaggle.com/datasets/tanreinama/900k-diffusion-prompts-dataset?select=diffusion_prompts.csv) 모델들을 각각 1000장, 2000장, 2000장 뽑아서 추가하였고(어떤 생성모델을 사용하였는지 몰랐기에, 되도록 고르게 사용하였고, Dalle Dataset의 퀄리티가 좋지 못해 비중 축소.), Image Net에서 실제 데이터를 5000장 추가 하여 데이터 증강을 진행하였습니다(Train set에서 Fake : real 비율이 1:1).
<!-- 
![fake_01956](https://github.com/taemin-steve/Fake_or_Real/assets/75752289/12402553-9f8a-46d4-ad52-87c0617a4238)
![SDfake_5276](https://github.com/taemin-steve/Fake_or_Real/assets/75752289/15b1ca3d-162b-4218-9af8-a7ce1858fd09)
![fakeimage_655](https://github.com/taemin-steve/Fake_or_Real/assets/75752289/cd412610-4b9b-4b10-a388-102a62eb8457)
-->

<div style="width: 33.33%; float: left;">
    <img src="https://github.com/taemin-steve/Fake_or_Real/assets/75752289/12402553-9f8a-46d4-ad52-87c0617a4238" alt="Image 1">
</div>
<div style="width: 33.33%; float: left;">
    <img src="https://github.com/taemin-steve/Fake_or_Real/assets/75752289/15b1ca3d-162b-4218-9af8-a7ce1858fd09" alt="Image 2">
</div>
<div style="width: 33.33%; float: left;">
    <img src="https://github.com/taemin-steve/Fake_or_Real/assets/75752289/cd412610-4b9b-4b10-a388-102a62eb8457" alt="Image 3">
</div>
<div style="clear: both;"></div>

##### 2) torchvisio.transforms
Torchvision의 transforms function을 활용하여 기존데이터와, 외부데이터를 추가로 증강하였습니다. Albumentations 모듈은 많이 사용해 보았기에 Torchvision 자체의 함수를 활용해 보았습니다. Filp, Rotation, ColorJitter, Crop등을 다양하게 테스트 해 보았고 결과적으로는 아래 3가지 증강을 활용하였습니다. 

```python 
self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
```

```python 
self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(200),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(30),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
```

```python 
self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(p = 1),
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
```
## _Ensemble_
최종적으로는 다양한 모델들에 대하여 ensemble을 soft vote 방식으로 진행하여 성능을 최대한 끌어 올렸습니다.
**Train Set의 이미지의 size가 다양했기에, 다양한 size**로 학습한 모델들을 앙상블 하면 더 좋은 성능을 얻을 것이라고 판단하였고, 이미지의 크기를 다르게 하여 학습한 모델들중 성능이 좋은 다섯개를 선택하여 앙상블을 진행하였습니다. (224x224, 256x256 , 456x456)
