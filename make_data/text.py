import torch
from torch.backends import cudnn

# CUDNN이 사용 가능한지 확인
if torch.cuda.is_available():
    # CUDNN 초기화
    cudnn.enabled = True
    cudnn.benchmark = True

    # 임의의 텐서 생성 및 GPU로 이동
    x = torch.randn(3, 3).cuda()

    # 연산 수행
    y = torch.matmul(x, x.t())

    # 결과 출력
    print(y)
else:
    print("CUDA가 사용할 수 없습니다.")