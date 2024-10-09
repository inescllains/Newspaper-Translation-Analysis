# Install required packages (uncomment the following lines if running in a new environment)
# !pip install googletrans==4.0.0-rc1
# !pip install news-please
# !pip install -U easynmt

from newsplease import NewsPlease
from easynmt import EasyNMT
from googletrans import Translator
import pandas as pd
import textwrap
from torch import cuda

# List of URLs to fetch articles
urls = [
    "https://www.publico.es/es/politica/podemos-completa-su-desconexion-de-sumar-y-toma-posicion-de-cara-a-la-investidura-de-sanchez/",
    "https://www.publico.es/es/politica/javier-de-andres-el-nuevo-lider-del-pp-vasco/",
    "https://www.publico.es/es/politica/belarra-teniamos-que-frenar-la-operacion-para-sustituir-a-podemos-por-una-izquierda-servil-al-regimen/",
    "https://www.publico.es/es/politica/consignas-franquistas-y-gritos-de-sanchez-a-prision-frente-a-la-sede-del-psoe-en-madrid/",
    "https://www.publico.es/es/politica/podemos-e-iu-en-alerta-por-el-reparto-ministerial-entre-pedro-sanchez-y-yolanda-diaz/",
    "https://www.publico.es/es/politica/la-fecha-de-investidura-a-la-espera-de-puigdemont/",
    "https://www.publico.es/es/politica/la-figura-del-verificador-entre-el-psoe-y-erc-un-primer-paso-hacia-la-supervision-de-las-negociaciones-del-conflicto-catalan/",
    "https://www.publico.es/es/politica/podemos-abre-una-nueva-etapa-para-desligarse-politica-y-estrategicamente-de-sumar/",
    "https://www.publico.es/es/politica/un-juzgado-de-madrid-estudiara-el-13-de-noviembre-una-medida-cautelar-de-paralizar-la-tramitacion-de-la-amnistia/",
    "https://www.publico.es/es/politica/la-audiencia-nacional-abre-juicio-por-terrorismo-contra-doce-miembros-de-los-cdr/"
]

# Retrieve articles from the URLs
articles = []
for url in urls:
    article = NewsPlease.from_url(url)
    articles.append(article)

# Prepare texts and titles for translation
for_translation = [article.maintext for article in articles]
titles = [article.title for article in articles]

# Check for GPU acceleration
device = 'cuda' if cuda.is_available() else None
print("Using device:", device)

# Translate using EasyNMT
model = EasyNMT('m2m_100_1.2B', device=device)
translated_texts = []
for t in model.translate_stream(for_translation, target_lang="en", show_progress_bar=True, batch_size=5):
    translated_texts.append(t)

# Save translated texts to DataFrame for EasyNMT
translated_df = pd.DataFrame(list(zip(titles, for_translation, translated_texts, urls)),
                             columns=['title', 'original_text', 'facebook_m2m', 'url'])
translated_df.to_csv("translated_facebook.csv", index=False)

# Translate using Google Translate
translator = Translator()
df_google = pd.DataFrame(columns=['title', 'original_text', 'google_translate', 'url'])

# Loop through each URL in the list
for url in urls:
    article = NewsPlease.from_url(url)

    # Extract title and main text
    title = article.title
    text = article.maintext

    # Split the text into smaller chunks
    text_split = textwrap.wrap(text, 500)
    translated = []

    # Translate each chunk
    for t in text_split:
        translation = translator.translate(t, dest='en', src='es')
        translated.append(translation.text)

    # Join the translated chunks into a single string
    translated_final = ' '.join(translated)

    # Append data to DataFrame
    df_google = df_google.append({'title': title, 'original_text': text, 'google_translate': translated_final, 'url': url}, ignore_index=True)

# Save Google Translate results to CSV
df_google.to_csv("translated_google.csv", index=False)

# Merge the three dataframes
df_facebook = pd.read_csv("translated_facebook.csv")
df_opus = pd.read_csv("translated_opus.csv")  # Make sure you have the translated_opus.csv file
df_merged_1 = df_google.merge(df_opus, on=["title", "original_text", "url"], how="inner")
df_merged_final = df_merged_1.merge(df_facebook, on=["title", "original_text", "url"], how="inner")

# Drop unnecessary columns and save the final dataframe
df_merged_final.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y', 'Unnamed: 0'], inplace=True, errors='ignore')
df_final = df_merged_final[['title', 'original_text', 'google_translate', 'opus-mt', 'facebook_m2m', 'url']]
df_final.to_csv("final_dataset_lains.csv", index=False)

print("Translation and merging completed successfully!")
