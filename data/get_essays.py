'''
Script to scrape all essays from Paul Graham's website. 
Output is saved as 'pg_essays.json'. 

'''


import sys, json
import requests
from bs4 import BeautifulSoup


def scrape_paul_graham_essays(url):
    """
    Scrapes the titles and contents of Paul Graham's essays from the given URL.

    Args:
        url (str): The URL of the Paul Graham's articles page.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'title', 'url' and 'content' of an essay. Returns an empty list
              if scraping fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')

        essay_list = []
        links = soup.find_all('a')

        for link in links:
            href = link.get('href')
            title = link.text.strip()
            if href and href.endswith('.html') and href != 'index.html':
                essay_url = f"https://paulgraham.com/{href}"
                essay_content = scrape_essay_content(essay_url)
                if essay_content:
                    essay_list.append({'title': title, 'url': essay_url, 'content': essay_content})	

        return essay_list

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def scrape_essay_content(url):
    """
    Scrapes the main content of a single Paul Graham essay page.

    Args:
        url (str): The URL of the individual essay.

    Returns:
        str: The main content of the essay, or None if scraping fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        main_text = soup.find('font', {'size': '2'}).get_text(separator="\n", strip=True)

        return main_text


        sys.exit()
        essay_text = ""
        for p in content_paragraphs:
            font_tag = p.find('font')
            if font_tag and font_tag.has_attr('size') and font_tag['size'] == '+1':
                essay_text += font_tag.text.strip() + "\n\n"

        return essay_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error during request for {url}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return None

if __name__ == "__main__":
    articles_url = "https://paulgraham.com/articles.html"
    essays = scrape_paul_graham_essays(articles_url)

    if essays:
        print(f"Successfully scraped {len(essays)} essays:")
        for essay in essays:
            print(f"\nTitle: {essay['title']}")
            # You can uncomment the next line to see the full content (it will be long)
            # print(f"Content:\n{essay['content'][:500]}...") # Print first 500 characters
    else:
        print("Failed to scrape essays.")

    with open('test_pg_essays.json', 'w', encoding='utf-8') as f:
        json.dump(essays, f, indent=4)  # indent for better readability
    print("Essays saved to 'pg_essays.json'")    
