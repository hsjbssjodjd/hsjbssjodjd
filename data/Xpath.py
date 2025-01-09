#用来提取bug报告，包含URL、Summary、Description属性保存在json文件内
import pandas as pd
import requests
from lxml import html
import os
import json


def extract_content(url, summary_xpath, description_xpath):
    """
    Extract Summary, Description, and include URL from the given webpage using the specified XPaths.
    """
    try:
        # Fetch the content of the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the HTML content
        tree = html.fromstring(response.content)

        # Extract Summary and Description
        summary = tree.xpath(summary_xpath)
        description = tree.xpath(description_xpath)

        # Get the text content or use a default message if not found
        summary_text = summary[0].text_content().strip() if summary else "No Summary Found"
        description_text = description[0].text_content().strip() if description else "No Description Found"

        return {"URL": url, "Summary": summary_text, "Description": description_text}
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return {"URL": url, "Summary": "Error", "Description": "Error"}


def save_to_json(output_dir, index, content):
    """
    Save the extracted content (including URL) to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"bug_report_{index}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    print(f"Saved content to {file_path}")


def process_csv_and_extract(csv_file, output_dir, summary_xpath, description_xpath):
    """
    Read the last column of a CSV file, extract URLs, and crawl each URL for content
    matching the specified XPaths. Save the results to individual JSON files.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        if df.empty or df.columns.size == 0:
            print("CSV file is empty or malformed.")
            return

        # Get the last column (assumes the last column contains URLs)
        urls = df.iloc[:, -1]

        for index, url in enumerate(urls, start=1):
            print(f"Processing URL {index}: {url}")
            content = extract_content(url, summary_xpath, description_xpath)
            save_to_json(output_dir, index, content)

        print(f"All URLs processed. Files saved in {output_dir}")
    except Exception as e:
        print(f"Error processing CSV file: {e}")


if __name__ == "__main__":
    # Input CSV file
    csv_file = "/media/oscar6/6F682A90B86D8F9F/wkb/FaultLocSim/VerilatorRepository.csv"  # Replace with your CSV file path

    # Output directory
    output_dir = "Verilator/bug_report"  # Directory to save files

    # Specify the XPaths for Summary and Description
    summary_xpath = '//*[@id="partial-discussion-header"]/div[1]/div/h1/bdi'
    description_xpath = '//*[starts-with(@id, "issue-")]/div/div[2]/task-lists/table/tbody/tr[1]'

    # Run the script
    process_csv_and_extract(csv_file, output_dir, summary_xpath, description_xpath)
