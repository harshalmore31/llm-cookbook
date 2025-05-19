from crawlee import playwright_crawler, Request
from playwright.sync_api import Page
from bs4 import BeautifulSoup
import re
import gradio as gr
import asyncio
import json


async def scrape_models(url="https://docs.litellm.ai/"):
   """
   Crawls and scrapes litellm documentation website for supported models
   using crawlee and playwright

    Args:
        url: The URL of the litellm documentation.

    Returns:
      A list of strings containing supported models
   """
   async def request_handler(page: Page, request: Request, playwright_crawler: playwright_crawler):
       # Find all the links on the page
        links = await page.locator("a").evaluate_all(
           "(elements) => elements.map(element => element.getAttribute('href'))"
        )
        model_links = []
        for link in links:
            if "models" in link:
                model_links.append(link)

        models = []
        for link in model_links:
            try:
                await page.goto(f"{url}{link}")
                model_list_items = await page.locator("li").all()

                for item in model_list_items:
                    item_text = await item.inner_text()

                    model_pattern = r"([a-zA-Z0-9-]+(?:/[a-zA-Z0-9.-]+)?)"
                    matches = re.findall(model_pattern, item_text)
                    if matches:
                        models.extend(matches)


            except Exception as e:
                print(f"Error processing {link}: {e}")

        # Remove duplicates
        unique_models = []
        seen = set()
        for model in models:
            if model not in seen:
                unique_models.append(model)
                seen.add(model)


        await playwright_crawler.context.close()
        return unique_models

   crawler = playwright_crawler(
        request_handler=request_handler
    )
   await crawler.run([url])
   return await crawler.get_results()



def gradio_app(models):
    """
    Creates a Gradio app for selecting a model.

    Args:
        models: A list of strings representing supported models

    Returns:
        A Gradio Interface object
    """
    if models:
        models_list = list(set(models[0].output)) # The result from crawlee is a list of dictionaries that have as output the list of models
    else:
      models_list = ["No models found"]

    with gr.Blocks() as demo:
            gr.Markdown("## LiteLLM Model Selector")
            dropdown = gr.Dropdown(
                choices=models_list,
                label="Select a LiteLLM Model",
            )
    return demo




async def main():
    """
    Scrapes models from the litellm website and creates a Gradio app
    """
    models = await scrape_models()
    demo = gradio_app(models)
    demo.launch()

if __name__ == "__main__":
    asyncio.run(main())