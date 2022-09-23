# 1. ToTensor
# ToTensor는 매우 일반적으로 사용되는 conversion transform입니다. 
# PyTorch에서 우리는 주로 텐서 형태의 데이터로 작업합니다. 
# 입력 데이터가 NumPy 배열 또는 PIL 이미지 형식인 경우 ToTensor를 
# 사용하여 텐서 형식으로 변환할 수 있습니다. transforms.ToTensor()

# 2. Normalize
# Normalize 작업은 텐서를 가져와 평균 및 표준 편차로 정규화합니다. 
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) - mean : (sequence)형식으로 평균을 입력하며, 
# 괄호 안에 들어가 있는 수의 개수가 채널의 수이다. - std : (sequence)형식으로 표준을 입력하며, 
# 마찬가지로 괄호 안에 들어가 있는 수의 개수가 채널의 수이다.

# 3. CenterCrop
# 이것은 중앙에서 주어진 텐서 이미지를 자릅니다. transform.CenterCrop(height, width) 
# 형식으로 자르려는 크기를 입력으로 제공할 수 있습니다.

# 4. RandomHorizontalFlip
# 이 변환은 주어진 확률로 이미지를 무작위로 수평으로 뒤집을(flip) 것입니다. 
# 이 확률은 매개변수 'p'를 통해 설정할 수 있습니다. p의 기본값은 0.5입니다.

# 5. RandomRotation
# 이 변환은 이미지를 각도만큼 무작위로 회전합니다. 
# 도(degree) 단위의 각도는 해당 매개변수 "degree"에 대한 입력으로 제공될 수 있습니다.

# 6. Grayscale
# 이 변환은 원본 RGB 이미지를 회색조로 변경합니다. 
# " num_output_channels" 매개변수에 입력으로 원하는 채널 수를 제공할 수 있습니다 .

# 7. 가우시안 블러
# 여기에서 이미지는 무작위로 선택된 가우시안 흐림 효과로 흐려집니다. 
# kernel_size 인수를 제공하는 것은 필수입니다.

# 8. RandomApply
# 이 변환은 확률로 주어진 transformation 들을 무작위로 적용합니다.

# 9. Resize 
# shape를 조절할 수 있다.
# ex) torch.Size([1, 250, 250]) =>Resize((28,28))=> torch.Size([1, 28, 28])
# transform = transforms.RandomApply([transforms.RandomSizedCrop(200),transforms.RandomHorizontalFlip()],p=0.6)
# tensor_img = transform(image)
# 9. Compose
# transform에 여러 단계가 있는 경우, Compose를 통해 여러 단계를 하나로 묶을 수 있습니다. transforms에 속한 함수들을 Compose를 통해 묶어서 한번에 처리할 수 있습니다.

# transforms.Compose([ 
#    transforms.ToTensor(), 
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
# ])