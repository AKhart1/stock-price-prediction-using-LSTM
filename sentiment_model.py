import requests
from bs4 import BeautifulSoup

# # # Sentiment Model

# # Sentimental Analysis
def scrape_yahoo_nvda_news():
    url = "https://finance.yahoo.com/quote/NVDA/news?p=NVDA"  # NVIDIA news page
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # print(soup.prettify())
    
    news_list = []
    for item in soup.find_all("div", class_="Ov(h) Pend(44px) Pstart(25px)"):
        title = item.find("h3")
        link = item.find("a", href=True)
        
        if title and link:
            news_list.append({
                "title": title.text,
                "link": f"https://finance.yahoo.com{link['href']}"
            })

    return news_list

news = scrape_yahoo_nvda_news()

for article in news:
    print("Title:", article["title"])
    print("Link:", article["link"])
    print()