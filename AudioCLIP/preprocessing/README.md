1. 다운받은 vggsound video dataset에 대해 **extract_audio.py** 돌려서 audio 추출
2. 추출한 audio에 대해 **segment_audio.py** 돌려서 2초 segment 생성
3. 생성한 segments에 대해 **extract_label.py** 돌려서 text label 추출
4. 2.에서 생성된 segments와 가 3에서 생성된 text label이 train.py에서 사용되는 input
