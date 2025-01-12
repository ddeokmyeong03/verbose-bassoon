import cv2
import time

# 카메라 입력 시작
cap = cv2.VideoCapture(0)
start_time = None
signal_triggered = False

# 첫 프레임 설정 (배경으로 사용)
ret, first_frame = cap.read()
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임 전처리 (그레이스케일, 블러)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # 프레임 간 차이 계산
    frame_delta = cv2.absdiff(first_frame, gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 윤곽선 찾기 (움직임이 감지된 영역)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 특정 크기 이상의 움직임 감지 시
    detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 5000:  # 감지할 객체 크기 조절
            detected = True
            break

    # 감지된 시간 체크
    if detected:
        if start_time is None:
            start_time = time.time()
            print("감지 시작")
        elif time.time() - start_time >= 2 and not signal_triggered:
            print("신호 출력")  # 2초 이상 지속 감지된 경우
            signal_triggered = True
    else:
        start_time = None
        signal_triggered = False

    # 화면에 영상 출력
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
