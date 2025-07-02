import bchlib
import numpy as np
import torch  # PyTorch 사용 (sum, mean 계산을 위해)

def bch_error_correction_batch(gt_messages, pred_messages, bch_polynomial=8219, bch_bits=14):
    """
    BCH 오류 정정 함수 (배치 입력 지원)
    - gt_messages : 원본 메시지 (배치 형태 [batch, 32])
    - pred_messages : 예측된 메시지 (배치 형태 [batch, 32])
    
    반환값:
    - 정정된 메시지 (배치 형태 [batch, 32])
    - 오류 수정 전 정확도 (batch 평균)
    - 오류 수정 후 정확도 (batch 평균)
    """
    
    # BCH 객체 생성
    gt_messages = gt_messages.to(torch.uint8) 
    pred_messages = pred_messages.to(torch.uint8)
    bch = bchlib.BCH(bch_bits, bch_polynomial, swap_bits=False)
    
    batch_size = gt_messages.shape[0]  # 배치 크기 (16개 샘플)
    corrected_bits = []
    pred_accuracies = []
    corrected_accuracies = []
    

    for i in range(batch_size):
        # 개별 샘플에 대해 BCH 인코딩 & 오류 정정 수행
        gt_message = gt_messages[i].tolist()
        pred_message = pred_messages[i].tolist()
        
        # BCH 인코딩
        gt_message_byte = bytes(np.packbits(gt_message, bitorder='little'))
        encoded_byte = bch.encode(gt_message_byte)

        # 예측 메시지를 Byte 변환
        pred_message_byte = bytes(np.packbits(pred_message, bitorder='little'))
        corrupted_message_byte = bytearray(pred_message_byte)
        
        # 송신 데이터 생성 (예측 메시지 + BCH 패리티)
        data_byte = corrupted_message_byte + bytearray(encoded_byte)
        
        # === BCH 디코딩 ===
        corrupted_data = data_byte[:-bch.ecc_bytes]
        corrupted_ecc = data_byte[-bch.ecc_bytes:]
        bch.decode(corrupted_data, corrupted_ecc)
        
        # 오류 정정 수행
        corrected_data = bytearray(corrupted_data)
        corrected_ecc = bytearray(corrupted_ecc)
        bch.correct(corrected_data, corrected_ecc)
        
        # 원본 & 수정된 데이터를 비트 배열로 변환
        gt_bit_clean = np.unpackbits(np.frombuffer(gt_message_byte, dtype=np.uint8), bitorder='little')[:32]
        pred_bit_corrupted = np.unpackbits(np.frombuffer(corrupted_message_byte, dtype=np.uint8), bitorder='little')[:32]
        corrected_bit = np.unpackbits(np.frombuffer(corrected_data, dtype=np.uint8), bitorder='little')[:32]
        
        # 정확도 계산
        pred_acc = (gt_bit_clean == pred_bit_corrupted).sum() / 32  # 오류 정정 전
        corrected_acc = (gt_bit_clean == corrected_bit).sum() / 32  # 오류 정정 후
        
        corrected_bits.append(corrected_bit)
        pred_accuracies.append(pred_acc)
        corrected_accuracies.append(corrected_acc)
    
    # PyTorch 텐서 변환 (배치 형태 유지)
    corrected_bits = torch.tensor(np.array(corrected_bits), dtype=torch.float32)  # [batch, 32]
    pred_accuracies = torch.tensor(pred_accuracies).mean()  # 배치 평균
    corrected_accuracies = torch.tensor(corrected_accuracies).mean()  # 배치 평균
    
    return corrected_bits, pred_accuracies.item(), corrected_accuracies.item()

# === 테스트 코드 ===
# gt_messages = torch.randint(0, 2, (16, 32))  # [16, 32] 랜덤 GT 메시지 생성
# pred_messages = gt_messages.clone()  # 복사 후 일부 에러 추가
# pred_messages[0, 5] = 1 - pred_messages[0, 5]  # 일부 비트 플립
# pred_messages[3, 10] = 1 - pred_messages[3, 10]

# corrected_bits, pred_acc, corrected_acc = bch_error_correction_batch(gt_messages, pred_messages)

# print(f"🔹 배치 예측 메시지 평균 정확도: {pred_acc:.2f}%")
# print(f"🔹 배치 오류 수정 후 평균 정확도: {corrected_acc:.2f}%")
# print(f"🔹 정정된 메시지 (샘플 1개): {corrected_bits[0].tolist()}")
