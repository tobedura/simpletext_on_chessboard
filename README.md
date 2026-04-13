# simpletext_on_chessboard

체스보드 영상에서 카메라 포즈를 추정하고, AR로 텍스트를 체스보드 위에 표시하는 프로그램입니다.

## 기능
- 체스보드 코너 검출 (`cv2.findChessboardCorners`)
- 카메라 포즈 추정 (`cv2.solvePnP`)
- 체스보드 평면 위에 "BOARD" 텍스트를 AR로 렌더링
- 카메라 위치(XYZ) 실시간 표시

## 데모

![demo](assets/simpletext.gif)

## 실행 방법
```bash
python main.py
```
- `Space`: 일시정지
- `ESC`: 종료
