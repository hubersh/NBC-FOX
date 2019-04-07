#!/usr/bin/python3

"""@author Hunter Hubers
   @Created 2019-03-24
"""

import re
import requests
import sys

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
        """Clean the text from a web page

        :return: Cleaned web page text data
        """
        output = []
        for item in self.data:
            for content in item:
                if content == '':
                    continue
                elif len(content.split()) < 5:
                    continue
                else:
                    output.append(content)
        return output

    @staticmethod
    def export_list_data(filename, data):
        """Export the text data from a web page to a file

        :param filename: Name of the file to write to
        :param data: Data to be written to the file
        :return: None
        """
        with open(filename, 'w+') as F:
            F.write(str(data)+"\n")


def external_call(url):
    """Create a temporary file for use in classifier.py

    :param url: web page to scrape
    :return: None
    """

    live = WebScrape()
    live.get_page(url)
    live.set_data()

    output = live.clean_content()
    with open("temp.txt", "w+", encoding="utf-8") as temp:
        for string in output:
            temp.write(string+"\n")
    print("Data Pulled")


if __name__ == '__main__':

    external_call('https://www.foxnews.com/politics/obama-warns-democrats-against-creating-circular-firing-squad')
    # Initialize the class
    # S = WebScrape()
    #
    # # Pull the page data
    # S.get_page(sys.argv[1])
    # S.set_data()

    # Clean and export the data
    # text_data = S.clean_content()
    # S.export_list_data('output3.txt', text_data)

