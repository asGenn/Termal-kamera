import os
import pandas as pd
import base64
from PIL import Image
import io


def resim_base64_kodla(img_path):
    """Resmi base64 formatına çevirir"""
    try:
        with Image.open(img_path) as img:
            # Resmi belleğe kaydet
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            # Base64'e çevir
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Hata: {img_path} yüklenemedi - {str(e)}")
        return None


def olasilik_renk_kodu(prob):
    """Olasılık değerine göre renk kodu döndürür"""
    if prob >= 0.7:
        return "danger"
    elif prob >= 0.4:
        return "warning"
    else:
        return "info"


def risk_tipi_stil(risk_tipi):
    """Risk tipine göre stil bilgilerini döndürür"""
    stiller = {
        "Goreceli_Risk": {
            "renk": "info",
            "metin": "Göreceli Risk",
            "badge_class": "risk-type-Goreceli"
        },
        "Yuksek_Risk": {
            "renk": "warning",
            "metin": "Yüksek Risk",
            "badge_class": "risk-type-Yuksek"
        },
        "Cok_Yuksek_Risk": {
            "renk": "danger",
            "metin": "Çok Yüksek Risk",
            "badge_class": "risk-type-Cok-Yuksek"
        }
    }
    return stiller.get(risk_tipi, {})


def html_olustur(df):
    """DataFrame'den HTML oluşturur"""

    # İstatistikleri hesapla
    toplam_yanlis = len(df)
    risk_dagilimi = df['Gerçek Etiket'].value_counts()

    # Risk dağılımı HTML'ini oluştur
    risk_dagilimi_html = ""
    for risk_tip, sayi in risk_dagilimi.items():
        stil = risk_tipi_stil(risk_tip)
        risk_dagilimi_html += f"""
            <div class="me-4">
                <span class="badge bg-{stil['renk']} fs-6">{stil['metin']}</span>
                <span class="fs-5 ms-2">{sayi}</span>
            </div>
        """

    html_template = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Yanlış Tahmin Analizi</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f8f9fa; }}
            .card {{
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: transform 0.3s;
                border: none;
                border-radius: 10px;
            }}
            .card:hover {{
                transform: translateY(-5px);
            }}
            .probability-badge {{
                margin: 2px;
                font-size: 0.9em;
            }}
            .wrong-prediction {{
                color: #dc3545;
                font-weight: bold;
            }}
            .card-img-top {{
                height: 300px;
                object-fit: contain;
                background-color: #fff;
                padding: 10px;
                border-radius: 10px 10px 0 0;
            }}
            .max-probability {{
                font-weight: bold;
                border: 2px solid currentColor;
            }}
            .risk-type-badge {{
                position: absolute;
                top: 10px;
                right: 10px;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 0.8em;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 0.9);
            }}
            .risk-type-Goreceli {{
                color: #0dcaf0;
                border: 2px solid #0dcaf0;
            }}
            .risk-type-Yuksek {{
                color: #ffc107;
                border: 2px solid #ffc107;
            }}
            .risk-type-Cok-Yuksek {{
                color: #dc3545;
                border: 2px solid #dc3545;
            }}
            .stats-container {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .stats-title {{
                color: #6c757d;
                font-size: 0.9em;
                margin-bottom: 10px;
            }}
            .filter-btn {{
                margin: 5px;
                border-radius: 20px;
                min-width: 120px;
            }}
            .filter-btn.active {{
                background-color: #0d6efd;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid py-4">
            <h1 class="text-center mb-4">Yanlış Tahmin Analizi</h1>
            
            <!-- İstatistikler -->
            <div class="stats-container mb-4">
                <div class="row">
                    <div class="col-md-4">
                        <div class="stats-title">Toplam Yanlış Tahmin</div>
                        <h3>{toplam_yanlis}</h3>
                    </div>
                    <div class="col-md-8">
                        <div class="stats-title">Risk Tiplerine Göre Dağılım</div>
                        <div class="d-flex flex-wrap">
                            {risk_dagilimi_html}
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Filtreler -->
            <div class="text-center mb-4">
                <button class="btn btn-outline-primary filter-btn active" data-risk="all">Tümü</button>
                <button class="btn btn-outline-info filter-btn" data-risk="Goreceli_Risk">Göreceli Risk</button>
                <button class="btn btn-outline-warning filter-btn" data-risk="Yuksek_Risk">Yüksek Risk</button>
                <button class="btn btn-outline-danger filter-btn" data-risk="Cok_Yuksek_Risk">Çok Yüksek Risk</button>
            </div>

            <div class="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4">
    """

    for _, row in df.iterrows():
        img_path = os.path.join("Basınç Yarası Riski",
                                row['Gerçek Etiket'], row['Dosya Adı'])
        img_base64 = resim_base64_kodla(img_path)

        if img_base64 is None:
            continue

        # Olasılıkları hesapla
        probs = {
            'Göreceli': row['Göreceli Risk Olasılığı'],
            'Yüksek': row['Yüksek Risk Olasılığı'],
            'Çok Yüksek': row['Çok Yüksek Risk Olasılığı']
        }
        max_prob = max(probs.values())

        stil = risk_tipi_stil(row['Gerçek Etiket'])

        html_template += f"""
                <div class="col risk-item" data-risk="{row['Gerçek Etiket']}">
                    <div class="card h-100">
                        <div class="position-relative">
                            <img src="{img_base64}" class="card-img-top" alt="{row['Dosya Adı']}">
                            <span class="risk-type-badge {stil['badge_class']}">{stil['metin']}</span>
                        </div>
                        <div class="card-body">
                            <h5 class="card-title">{row['Dosya Adı']}</h5>
                            <p class="card-text">
                                <strong>Gerçek:</strong> {stil['metin']}<br>
                                <strong>Tahmin:</strong> <span class="wrong-prediction">{row['Tahmin'].replace('_', ' ')}</span>
                            </p>
                            <div class="probability-container">
                                <strong>Olasılıklar:</strong><br>
        """

        for label, prob in probs.items():
            max_class = "max-probability" if prob == max_prob else ""
            html_template += f"""
                                <span class="badge bg-{olasilik_renk_kodu(prob)} probability-badge {max_class}">
                                    {label}: {prob:.2f}
                                </span>
            """

        html_template += """
                            </div>
                        </div>
                    </div>
                </div>
        """

    html_template += """
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Risk tipine göre filtreleme
            document.querySelectorAll('.filter-btn').forEach(button => {
                button.addEventListener('click', function() {
                    // Aktif butonu güncelle
                    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
                    this.classList.add('active');
                    
                    const risk = this.getAttribute('data-risk');
                    document.querySelectorAll('.risk-item').forEach(item => {
                        if (risk === 'all' || item.getAttribute('data-risk') === risk) {
                            item.style.display = '';
                        } else {
                            item.style.display = 'none';
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """

    return html_template


def tum_yanlis_tahminleri_goster():
    """Tüm yanlış tahminleri HTML sayfasında gösterir"""

    # Detaylı tahmin raporunu oku
    df = pd.read_csv("analysis_results/detayli_tahmin_raporu.csv")

    # Sadece yanlış tahminleri al
    yanlis_tahminler = df[df['Doğru mu?'] == False].copy()

    if len(yanlis_tahminler) == 0:
        print("Hiç yanlış tahmin bulunamadı.")
        return

    # HTML oluştur
    html_content = html_olustur(yanlis_tahminler)

    # HTML dosyasını kaydet
    output_path = "raporlar/yanlis_tahminler_raporu.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Rapor oluşturuldu: {output_path}")

    # HTML dosyasını otomatik aç
    os.system(f'start {output_path}')


if __name__ == "__main__":
    print("Yanlış tahminler raporu hazırlanıyor...")
    tum_yanlis_tahminleri_goster()
