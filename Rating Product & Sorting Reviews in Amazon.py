
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı



###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
import pandas as pd

import math

import scipy.stats as st

from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("DERSLER/MEASUREMENT PROBLEMS/case study-rating product sorting review amazon/Rating Product&SortingReviewsinAmazon/amazon_review.csv")
df.head()
df["overall"].mean() #4.5875
###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################
df.describe().T

def time_weighted_average(dataframe,w1=30,w2=27,w3=23,w4=20):
    return dataframe.loc[(dataframe["day_diff"]>0) & (dataframe["day_diff"]<180),"overall"].mean() * w1/100 + \
        dataframe.loc[(dataframe["day_diff"]>180) & (dataframe["day_diff"]<365),"overall"].mean() * w2/100 + \
        dataframe.loc[(dataframe["day_diff"]>365) & (dataframe["day_diff"]<650),"overall"].mean() * w3/100 + \
        dataframe.loc[(dataframe["day_diff"]>650) & (dataframe["day_diff"]<1000),"overall"].mean() * w4/100

time_weighted_average(df) #4.60579
###################################################
# Adım 3:  Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
###################################################
df.loc[(df["day_diff"]>0) & (df["day_diff"]<180),"overall"].mean()  #4.6936
df.loc[(df["day_diff"]>180) & (df["day_diff"]<365),"overall"].mean() #4.6813
df.loc[(df["day_diff"]>365) & (df["day_diff"]<650),"overall"].mean()  #4.5577
df.loc[(df["day_diff"]>650) & (df["day_diff"]<1000),"overall"].mean()  #4.4272
#ürün son zamanlarda daha yüksek puanlar almış.
###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################
df["helpful"].nunique()
df.head()
###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.tail(30)
###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################
def score_pos_neg_diff(up,down):
    return up-down
df["score_pos_neg_diff"] = df.apply(lambda x:score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]),axis=1)

def score_average_rating(up,down):
    if up ==0:
        return 0
    return up / (up+down)
df["score_average_rating"] = df.apply(lambda x:score_average_rating(x["helpful_yes"], x["helpful_no"]),axis=1)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
df["wilson_lower_bound"] = df.apply(lambda x:wilson_lower_bound(x["helpful_yes"], x["helpful_no"]),axis=1)

df.sort_values("helpful_yes",ascending=False).head(20)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound",ascending=False).head(20)

