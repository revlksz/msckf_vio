import re
import numpy as np
import matplotlib.pyplot as plt

def parse_vio_logs(file_path):
    frames = []
    fps_list = []
    features_list = []
    vio_pos = []
    gt_pos = []
    
    t_orb = []
    t_match = []
    t_msckf = []
    
    line1_regex = re.compile(r"(?:\[ZUPT\]\s*)?Frame\s+(\d+)\s*\|\s*FPS:\s*([\d.]+)\s*\|\s*Features:\s*(\d+)\s*\|\s*Pos:\s*\[\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\]\s*\|\s*GT:\s*\[\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\]")
    line2_regex = re.compile(r"\[Zamanlar\]\s*ORB:\s*([\d.]+)ms\s*\|\s*Match:\s*([\d.]+)ms\s*\|\s*MSCKF:\s*([\d.]+)ms")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Hata: '{file_path}' dosyası bulunamadı!")
        return None
        
    for i in range(0, len(lines) - 1, 2):
        l1 = lines[i].strip()
        l2 = lines[i+1].strip()
        
        m1 = line1_regex.search(l1)
        m2 = line2_regex.search(l2)
        
        if m1 and m2:
            frames.append(int(m1.group(1)))
            fps_list.append(float(m1.group(2)))
            features_list.append(int(m1.group(3)))
            vio_pos.append([float(m1.group(4)), float(m1.group(5)), float(m1.group(6))])
            gt_pos.append([float(m1.group(7)), float(m1.group(8)), float(m1.group(9))])
            
            t_orb.append(float(m2.group(1)))
            t_match.append(float(m2.group(2)))
            t_msckf.append(float(m2.group(3)))

    return (np.array(frames), np.array(fps_list), np.array(features_list), 
            np.array(vio_pos), np.array(gt_pos), 
            np.array(t_orb), np.array(t_match), np.array(t_msckf))

def main():
    # ── DOSYA İSİMLERİ ────────────────────────────────────────
    laptop_file = "vio_estimation_logs_laptop.txt"
    jetson_file = "vio_estimation_logs_jetson.txt"
    # ──────────────────────────────────────────────────────────
    
    print("Log dosyaları yükleniyor...")
    laptop_data = parse_vio_logs(laptop_file)
    jetson_data = parse_vio_logs(jetson_file)
    
    if laptop_data is None or jetson_data is None:
        print("Lütfen her iki log dosyasının da mevcut olduğundan emin olun.")
        return

    # Verileri ayıkla
    l_frames, l_fps, _, l_vio, gt, l_orb, l_match, l_msckf = laptop_data
    j_frames, j_fps, _, j_vio, _, j_orb, j_match, j_msckf = jetson_data

    # Grafik teması
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------
    # GRAFİK 1: 3D Trajectory Karşılaştırması (Laptop vs Jetson vs GT)
    # -------------------------------------------------------------
    fig1 = plt.figure(figsize=(11, 8))
    ax = fig1.add_subplot(111, projection='3d')
    
    # 3 Ayrı Çizgi (İstediğiniz gibi ayrı renklerde)
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'g-', label='Ground Truth (Gerçek Rota)', linewidth=2.5)
    ax.plot(l_vio[:, 0], l_vio[:, 1], l_vio[:, 2], 'b--', label='Laptop MSCKF VIO', linewidth=1.5)
    ax.plot(j_vio[:, 0], j_vio[:, 1], j_vio[:, 2], 'r:', label='Jetson MSCKF VIO', linewidth=1.5)
    
    # Başlangıç / Bitiş noktaları
    ax.scatter(gt[0, 0], gt[0, 1], gt[0, 2], c='black', marker='o', s=100, label='Başlangıç Noktası')
    ax.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], c='black', marker='X', s=100, label='Bitiş Noktası')
    
    ax.set_title("Donanım Karşılaştırmalı 3D Trajectory Analizi", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (Metre)")
    ax.set_ylabel("Y (Metre)")
    ax.set_zlabel("Z (Metre)")
    ax.legend(loc='upper left')
    
    # -------------------------------------------------------------
    # GRAFİK 2: Performans ve Hata Karşılaştırma Paneli
    # -------------------------------------------------------------
    fig2, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Sub-Plot 1: Mutlak Pozisyon Hataları (VIO vs GT)
    # Not: Eğer frame sayıları uyuşmuyorsa en kısa olan listeye göre hizalama yapılır
    min_len_l = min(len(gt), len(l_vio))
    min_len_j = min(len(gt), len(j_vio))
    l_errors = np.linalg.norm(gt[:min_len_l] - l_vio[:min_len_l], axis=1)
    j_errors = np.linalg.norm(gt[:min_len_j] - j_vio[:min_len_j], axis=1)
    
    axs[0].plot(l_frames[:min_len_l], l_errors, 'b-', label='Laptop Pozisyon Hatası (m)', alpha=0.8)
    axs[0].plot(j_frames[:min_len_j], j_errors, 'r-', label='Jetson Pozisyon Hatası (m)', alpha=0.8)
    axs[0].set_title("Zamana (Frame) Bağlı Donanım Performans Karşılaştırması", fontsize=14, fontweight='bold')
    axs[0].set_ylabel("Hata (Metre)")
    axs[0].legend(loc='upper left')
    
    # Sub-Plot 2: Toplam Algoritma Çalışma Süresi (ORB + Match + MSCKF)
    l_total_time = l_orb + l_match + l_msckf
    j_total_time = j_orb + j_match + j_msckf
    
    axs[1].plot(l_frames, l_total_time, 'b-', label='Laptop Toplam Çerçeve Süresi (ms)', alpha=0.7)
    axs[1].plot(j_frames, j_total_time, 'r-', label='Jetson Toplam Çerçeve Süresi (ms)', alpha=0.7)
    axs[1].set_ylabel("Süre (Milisaniye)")
    axs[1].legend(loc='upper left')
    
    # Sub-Plot 3: FPS Karşılaştırması
    axs[2].plot(l_frames, l_fps, 'b-', label='Laptop FPS', alpha=0.8)
    axs[2].plot(j_frames, j_fps, 'r-', label='Jetson FPS', alpha=0.8)
    axs[2].set_ylabel("FPS")
    axs[2].set_xlabel("Frame Numarası")
    axs[2].legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()