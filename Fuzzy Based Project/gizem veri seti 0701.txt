import openai
import numpy as np
import pandas as pd
import time  # time kütüphanesini ekliyoruz

# Azure OpenAI API bilgilerini ayarla
api_config = {
    "MAX_TOKENS": 8192,
    "model": "gpt-35-turbo",
    "AZURE_ENDPOINT": "https://xmew1-spoke0007-openai-13.openai.azure.com/",
    "API_VERSION": "2023-07-01-preview",
    "Key": "c12a556db7cb4151b2175c5378fc96bc",
    "DEPLOYMENT": "xmew1-spoke0007-openai-deployment-13-gpt3"
}

openai.api_type = "azure"
openai.api_base = api_config["AZURE_ENDPOINT"]
openai.api_version = api_config["API_VERSION"]
openai.api_key = api_config["Key"]

# Öğrenci verilerinin simülasyonu
num_students = 4000  # Toplam öğrenci sayısı
data = {
    "Student_ID": np.arange(1, num_students + 1),
    "Age": np.random.randint(18, 30, num_students),
    "Class_Level": np.random.choice(["Freshman", "Sophomore", "Junior", "Senior"], num_students),
    "Interest": np.random.choice(["Math", "Science", "Art", "History"], num_students),
    "Exam_Score": np.random.randint(50, 100, num_students),
    "Study_Hours": np.random.uniform(1, 10, num_students).round(2),
    "Engagement_Score": np.random.randint(1, 101, num_students),
    "Content_Viewed": np.random.randint(1, 50, num_students),
}

df = pd.DataFrame(data)

# GPT modelleriyle cümle üretimi
def generate_sentence(row, deployment):
    prompt = (f"A student interested in {row['Interest']} spends about {row['Study_Hours']} hours studying, "
              f"has an engagement score of {row['Engagement_Score']}, and viewed {row['Content_Viewed']} materials. "
              "Describe their learning behavior on the platform.")
    
    retry_count = 0
    while retry_count < 5:  # 5 kez tekrar dene
        try:
            response = openai.Completion.create(
                engine=deployment,  # Azure OpenAI için deployment adı kullanılıyor
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            return response["choices"][0]["text"].strip()
        except openai.error.RateLimitError:
            print("Rate limit exceeded, retrying in 2 seconds...")
            time.sleep(2)  # 2 saniye bekle
            retry_count += 1
    
    print("Rate limit exceeded after multiple retries.")
    return "Unable to generate description"  # Hata durumunda dönüş değeri

# Tüm öğrenci verileri için cümle üretimi
behavior_descriptions = []
request_count = 0  # İstek sayacını başlatıyoruz

for index, row in df.iterrows():
    if request_count >= 10:  # 10 istekten sonra bekle
        print("Max requests reached, sleeping for 30 seconds...")
        time.sleep(30)  # 30 saniye bekle
        request_count = 0  # Sayaç sıfırlanır
    
    behavior_descriptions.append(generate_sentence(row, api_config["DEPLOYMENT"]))
    request_count += 1

df["Behavior_Description"] = behavior_descriptions

# Veri setini kaydetme
df.to_csv("openai_generated_dataset.csv", index=False)
print("Veri seti oluşturuldu ve kaydedildi: openai_generated_dataset.csv")
