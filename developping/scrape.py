import requests
from bs4 import BeautifulSoup

# URL of the Litellm documentation homepage
url = 'https://docs.litellm.ai/docs/'

# Fetch the content of the homepage
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the links to the documentation pages
    doc_links = soup.find_all('a', href=True)
    doc_urls = [link['href'] for link in doc_links if 'docs.litellm' in link['href']]
    print(doc_urls)

    # Create a set to avoid duplicates and to store full URLs
    base_url = url
    full_urls = {base_url + link if link.startswith('/') else link for link in doc_urls}

    # Initialize a list to store the content
    documentation_content = []

    # Fetch and parse each documentation page
    for doc_url in full_urls:
        doc_response = requests.get(doc_url)
        if doc_response.status_code == 200:
            doc_soup = BeautifulSoup(doc_response.content, 'html.parser')
            # Extract the main content from the page
            main_content = doc_soup.find('main') or doc_soup.body
            if main_content:
                documentation_content.append(main_content.get_text())

    # Combine all content into a single string
    full_documentation = '\n\n'.join(documentation_content)

    # Save the documentation to a text file
    with open('litellm_documentation.txt', 'w', encoding='utf-8') as file:
        file.write(full_documentation)

    print("Documentation saved successfully.")
else:
    print(f"Failed to fetch the homepage. Status code: {response.status_code}")
