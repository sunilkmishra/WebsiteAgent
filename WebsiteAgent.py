import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import openai
from urllib.parse import urljoin, urlparse
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Set up WebDriver
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Get all links from a webpage
def get_all_links(driver, base_url, visited):
    links = set()
    try:
        driver.get(base_url)
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.TAG_NAME, "a")))
        hrefs = driver.execute_script("return Array.from(document.querySelectorAll('a'), a => a.getAttribute('href'));")
        parsed_base = urlparse(base_url)
        for href in hrefs:
            if href:
                absolute_url = urljoin(base_url, href)
                parsed = urlparse(absolute_url)
                if parsed.netloc == parsed_base.netloc:
                    links.add(absolute_url)
        return links
    except Exception as e:
        st.error(f"Error fetching links: {str(e)}")
        return set()

# Scrape webpage content
def scrape_page(driver, url):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        title = driver.title
        content = driver.find_element(By.TAG_NAME, "body").text
        return {"URL": url, "Title": title, "Content": content[:5000]}  # Limiting text to 5000 chars
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return None

# Recursive scraping function
def scrape_website(base_url, depth=3):
    visited = set()
    pages_data = []
    driver = setup_driver()
    
    def recursive_scrape(url, current_depth):
        if current_depth == 0 or url in visited:
            return
        visited.add(url)
        st.write(f"Scraping: {url}")
        page_data = scrape_page(driver, url)
        if page_data:
            pages_data.append(page_data)
            if current_depth > 1:
                new_links = get_all_links(driver, url, visited)
                for link in new_links:
                    recursive_scrape(link, current_depth - 1)
    
    recursive_scrape(base_url, depth)
    driver.quit()
    return pages_data

# Save data to CSV
def save_to_csv(pages_data, filename="website_data.csv"):
    if not pages_data:
        st.error("No data to save!")
        return None
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["URL", "Title", "Content"])
        writer.writeheader()
        writer.writerows(pages_data)
    return filename

# OpenAI Embeddings
def get_embedding(text):
    text = text.replace('\n', ' ')
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return response['data'][0]['embedding']

# Similarity function
def similar(page_embedding, question_embedding):
    return np.dot(page_embedding, question_embedding)

# Streamlit UI
st.title("Website Scraper & Q&A Bot")

# Scraping UI
url_input = st.text_input("Enter the website URL (e.g., https://example.com):")
scrape_depth = st.slider("Scraping Depth", 1, 5, 3)
scrape_btn = st.button("Start Scraping")

# Initialize session state for storing data
if "df" not in st.session_state:
    st.session_state.df = None

if scrape_btn:
    if not url_input.startswith("http"):
        url_input = "https://" + url_input
    with st.spinner("Scraping in progress..."):
        pages = scrape_website(url_input, scrape_depth)
        filename = save_to_csv(pages)

    if filename:
        st.success(f"Data saved to {filename}")
        df = pd.read_csv(filename)
        df['embeddings'] = df['Content'].apply(get_embedding)
        df.to_csv('./website_data_with_embeddings.csv', index=False)
        df.to_pickle('./website_data_with_embeddings.pkl')

        # Store the dataframe in session state
        st.session_state.df = df

        # Debugging message
        st.write(f"🔹 Loaded {len(df)} rows from CSV")

# Q&A UI
if st.session_state.df is not None:
    query = st.text_input("Ask a question about the website:")

    if query:
        question_embedding = get_embedding(query)
        st.session_state.df['distance'] = st.session_state.df['embeddings'].apply(lambda x: similar(x, question_embedding))
        st.session_state.df.sort_values('distance', ascending=False, inplace=True)

        if len(st.session_state.df) >= 3:
            context = (
                st.session_state.df.iloc[0]['Content'] + "\n" +
                st.session_state.df.iloc[1]['Content'] + "\n" +
                st.session_state.df.iloc[2]['Content']
            )
        else:
            context = "\n".join(st.session_state.df['Content'])

        if st.button("Get Answer"):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Use this information to answer the question: {query}\n\n{context}"}
                ],
                max_tokens=100,
                temperature=0.1
            )

            if response and "choices" in response and response.choices:
                st.write(f"**Answer:**\n{response.choices[0].message.content}")
            else:
                st.warning("No relevant information found. Try rephrasing your question.")
