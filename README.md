# audio_preprocessing
audio embedding을 text embedding dimension으로 변환


**✅ audio 전처리 과정 정의**

1. 전체 audio mel-spectrogram을 5개로 나누기 → 나눈 후에 label이 등장하지 않는 segment는 버리기
    ▶️ UnAV-100 dataset은 vggsound dataset과는 다르게 총 video의 길이가 모두 다릅니다. 따라서 아래 순서에 따라 전처리를 진행하였습니다.
   
    1) 원본 video로부터 audio 추출(raw audio)
    
    2) 하나의 raw audio(각각 다른 길이) → 10초 길이의 audio로 split
    
    - ex) 48s → 10s * 4 + 8s
      
    - 이때 위 예시에서 8초와 같이 10초가 되지 않는 분할된 audio에 대해서는 padding 적용 → 똑같이 10초로 맞춰줌
    
    3) 10s audio → 5개로 분할
    
    - 각 segment는 2초의 길이를 갖게됨
    
    4) annotation.json과 비교했을 때, 생성된 2초 segment 중에서 label에 해당하는 소리가 등장하지 않는 segment는 버리기
    
    - train 과정에서의 input은 label에 해당하는 소리가 등장하는 segment들만 가지고 학습
