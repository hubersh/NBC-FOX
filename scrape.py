#!/usr/bin/python3

"""@author Hunter Hubers
   @Created 2019-03-24
"""

import re
import requests

from bs4 import BeautifulSoup


class WebScrape:
    """Designed to scrape web pages"""
    def __init__(self):
        """Construct class variables"""
        self.data = []
        self.page = None
        self.soup = None

    def get_page(self, page):
        """Set the page and soup variables to a URL's contents

           :param page: A website URL
           :return: None
        """
        self.page = requests.get(page)
        # Create a BeautifulSoup object
        self.soup = BeautifulSoup(self.page.text, 'html.parser')

    def set_data(self):
        """Pull all of the text from a web page and store it in a class variable

           :return: None
        """
        self.data = [re.findall(r'(?<=>)([^{\n]*?)\w?(?=<)', str(self.soup))]

    def clean_content(self):
        """Clean the text from a web page"""
        output = []
        for item in self.data:
            for content in item:
                if content == '':
                    continue
                elif len(content.split()) < 5:
                    continue
                else:
                    output.append(content)
        self.data = output

    def format_data(self):
        """Format the data to an individual string.

           :return: String of all text data.
        """
        string_output = ""
        for string in self.data:
            string_output += string.strip()
        return string_output

    @staticmethod
    def export_list_data(filename, data):
        """Export the text data from a web page to a file

           :param filename: Name of the file to write to
           :param data: Data to be written to the file
           :return: None
        """
        with open(filename, 'w+', encoding="utf-8") as F:
            for string in data:
                F.write(string.strip()+" ")
            F.write("\n")


def external_call(url):
    """Create a temporary file for use in classifier.py

    :param url: The web page to scrape
    :return: Formatted string
    """
    live = WebScrape()
    live.get_page(url)
    live.set_data()
    live.clean_content()

    return live.format_data()


if __name__ == '__main__':

    external_call('https://www.foxnews.com/politics/obama-warns-democrats-against-creating-circular-firing-squad')

