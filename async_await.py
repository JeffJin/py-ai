import asyncio
import os
import aiohttp
import json

async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def read_example():
    url = 'http://example.com'
    content = await fetch_url(url)
    print(f"Content from {url}:\n{content[:300]}...")  # Print first 200 chars

def read_file():
    with open('./data/salary_Data.csv', 'r') as file:
        # lines = file.readlines()
        for line in file:
            print(line)


def write_json_file(filename, data):
    try:
        with open(filename, 'w') as file:
            json.dump(data, file)
    except IOError as e:
        print(f"Error writing to file: {e}")


def list_dir():
    entries = os.scandir('./data')
    for entry in entries:
        print(entry)

async def main():
    read_file()

if __name__ == "__main__":
    asyncio.run(main())