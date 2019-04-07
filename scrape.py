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
    def export_data(filename, data):
        """Export the text data from a web page to a file

        :param filename: Name of the file to write to
        :param data: Data to be written to the file
        :return: None
        """
        with open(filename, 'w+') as F:
            F.write(str(data)+"\n")


if __name__ == '__main__':

    # Initialize the class
    S = WebScrape()

    # Pull the page data
    S.get_page(sys.argv[1])
    S.set_data()

    # Clean and export the data
    text_data = S.clean_content()
    S.export_data('output3.txt', text_data)

