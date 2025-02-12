import pandas as pd

# CSV dosyasını oku (dosya adını uygun şekilde değiştirin)
csv_dosyasi = 'basinc_yarasi_model_predictions.csv'
df = pd.read_csv(csv_dosyasi)

# "Doğru mu?" sütunundaki değerler string ise, bunları boolean'a çeviriyoruz
if df["Doğru mu?"].dtype == object:
    df["Doğru mu?"] = df["Doğru mu?"].str.strip(
    ).str.lower().map({'true': True, 'false': False})

# Doğruluk oranını hesapla:
# Doğru tahmin sayısını toplam tahmin sayısına bölüp yüzdeye çeviriyoruz.
dogruluk_orani = df["Doğru mu?"].mean() * 100

print(f"Doğruluk Oranı: {dogruluk_orani:.2f}%")
