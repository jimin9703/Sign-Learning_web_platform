# 졸업프로젝트 Sign Learning (2022년 2학기) 

## 주제💻 딥러닝 기술을 활용한 수어교육 웹 플랫폼 개발_(한,영_지문자ver)

### 웹 플랫폼 구현 영상: 지문자 퀴즈풀이

<p align="center"><img src="https://user-images.githubusercontent.com/74172467/207249922-1b1d9703-8481-497a-9d17-de6e589ddf6d.gif " width= "700">

# 💫 Full demo on -> [Youtube](https://youtu.be/6CvHjVIJ6dA)💫

## ✏  서비스 기획과 전략

서비스의 전략과 방향을 얻기 위해 수어봉사동아리 '마주보다'와 인터뷰와 사전조사를 진행하였습니다.

마스크 착용으로 인하여 구어를 통한 의사소통에 어려움을 겪어 수어의 필요성과 수어교육의 수요가 늘어났음을 확인했습니다.

더하여 오프라인으로 수어를 배울 수 있는 교육기관이 매우 적고 영상이나 앱을 통한 교육은
일방향적인 정보 전달 뿐 본인이 수어를 옳게 또는 틀리게 하고 있는지 확인할 방법이 없다는 문제점이 있었습니다.


봉사 동아리와의 인터뷰|잠재적 사용자 파악|
|------|---|
<img src = "https://user-images.githubusercontent.com/74172467/206975572-868d47e1-be19-4aec-ac5c-cb985ad376c4.png" width="400" height= "300">| <img src = "https://user-images.githubusercontent.com/74172467/206976739-f2536cef-afa7-4c88-9974-a6babbfd0ec3.png" width="400">

## ✏ 교육 컨텐츠
첫 째, 학습페이지입니다.
왼쪽에 수어이미지를 보여주고 오른쪽에는 카메라 화면이 보여집니다.
현재 사용자가 표현하고 있는 수어를 실시간으로 확인할 수 있습니다.

둘 째, 퀴즈페이지입니다.
사용자는 주어진 지문자에 따라 카메라에 수어를 지어보이면 됩니다.
본인의 수어에 대한 피드백을 받을 수도 있고 배움에 대한 흥미도 올릴 수 있습니다.


학습페이지|퀴즈페이지|
|------|---|
<img src = "https://user-images.githubusercontent.com/74172467/206991804-e690ab10-97ca-4cc1-9407-da261862bed3.png" width="400" height= "300">| <img src = "https://user-images.githubusercontent.com/74172467/206992117-7576d8c0-dc44-40d6-b353-f6dd2bcaa22a.png" width="400">


## ✏ 서비스 디자인

지원하는 언어는 영어와 한국어이며 각 언어의 학습페이지와 시험페이지가 구현되어져 있습니다.

서비스 메인 화면|서비스 구조|
|------|---|
<img src = "https://user-images.githubusercontent.com/74172467/206986355-4f3f9861-9657-43f3-b09a-c1a65ef630fd.png" width="400" height= "300"> |<img src = "https://user-images.githubusercontent.com/74172467/206986048-e0ecb9e3-bfcb-46ce-9460-c8d41f2f29ac.png" width="400">

## 📌 기술적인 요소

첫 째로, 영상이미지 데이터를 기계학습에 활용할 수 있도록 벡터화하는 과정입니다.
이후 Mediapipe를 활용하여 손의 Keypoint를 찾고 이웃 keypoint와의 각도를 계산합니다. 
이 값을 정규화하여 csv파일로 저장합니다. 이렇게 dataset을 완성합니다.

Vetorization|degree calculation| saving as CSV file
|------|---|---|
|<img src = "https://user-images.githubusercontent.com/74172467/206989944-d3f0c22e-f98c-48f3-93c5-12220f8061de.png" width="300">| <img src = "https://user-images.githubusercontent.com/74172467/206990213-7e46d0b4-7684-4c3b-87f1-8a68deae22df.png" width="300">| <img src = "https://user-images.githubusercontent.com/74172467/206990338-967bf9c3-913c-4303-baff-878b975e55c6.png" width="300">|

## 📌 기술적인 요소 2 

처음에는 기계학습 학습해보았는데 정확도가 기대 이하였습니다.
LSTM을 사용했을 때에는 실시간적인 피드백이 어려웠습니다.

따라서 DNN모델을 선택하였고 과적합을 방지하기 위해 드랍아웃과 batch normalization을 사용하였습니다.

저희가 활용한 최종 학습모델은 7개의 Dense layer를 갖은 DNN으로 90초반의 정확도를 달성했습니다.

DNN Architeture|Loss rating & ACC score| Dropout & batch normalization
|------|---|---|
|<img src = "https://user-images.githubusercontent.com/74172467/206996563-d51cb975-a4a1-488d-9fdf-68d4e703a78b.png" width="300">| <img src = "https://user-images.githubusercontent.com/74172467/206996885-e6724dea-7236-43c4-adad-69cee6f3d92c.png" width="300">| <img src = "https://user-images.githubusercontent.com/74172467/206996969-8c3ca354-a9cd-4128-a1b5-f5ad009db5de.png" width="300">|







