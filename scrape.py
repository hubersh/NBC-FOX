#!/usr/bin/python3

import re
import requests
import sys

from bs4 import BeautifulSoup


class WebScrape:
    """Designed to scrape web pages."""
    def __init__(self, page):
        # TODO Add function docstring
        self.page = requests.get(page)
        # Create a BeautifulSoup object
        self.soup = BeautifulSoup(self.page.text, 'html.parser')

    def scrape_content(self):
        # TODO determine if divs is good enough
        paragraphs = self.soup.find_all('p')
        divs = self.soup.find_all('div')

        self.export_data('output.txt', divs)
        # TODO Loop though data and clear out unneeded divs
        # for text in divs:
        # TODO Implement regex cleaning
        #     print(text)

    def clean_content(self):
        # TODO Implement NLP cleaning
        pass

    @staticmethod
    def export_data(filename, data):
        # TODO Try to make method generic
        with open(filename, 'w+') as F:
            for text in data:
                F.write(str(text)+"\n")


if __name__ == '__main__':
    # Initialize the class
    S = WebScrape(sys.argv[1])
    S.scrape_content()

