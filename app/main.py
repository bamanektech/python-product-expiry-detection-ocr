from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import easyocr
import shutil
from pathlib import Path
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
import logging
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder ,SystemMessagePromptTemplate ,HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv


load_dotenv()


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()


# model=Ollama(model="mistral")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are a helpful OCR text extraction assistant for classifying expiry dates from the product packaging.
    I will provide you with the text extracted from the image, and you will classify the mfg date and expiry date from the text.
    If you can't find the mfg date and expiry date in the text, then you will say "can't detect the mfg date and expiry date from the Image".
    This type of text is usually found in product packaging.

    --- 
    🔖 All Label Variants:

    MFG (Manufacturing Date):
    - Mfg date, MFG:, mfg :, Packed on, Pckd on, MFG, Mfg, MFGD, MFGD:

    EXP (Expiry / Use By Date):
    - Exp date, Expire date, Exp:, Use by, Best Before, Use By, Best if used before, Better if used by

    Duration-based expiry:
    - Best Before XX YEARS
    - Use within 6 months of MFG
    - Shelf life: 18 months

    ---
    🗓 Date Format Variants:

    Slash-separated:  
    - DD/MM/YYYY, MM/DD/YYYY, YYYY/MM/DD  
    - DD/MON/YYYY, etc.

    Dash-separated:  
    - DD-MM-YYYY, MON-DD-YYYY, YYYY-MM-DD, etc.

    Dot-separated:  
    - DD.MM.YYYY, MM.DD.YYYY, etc.

    Alphanumeric:  
    - DDMMYYYY, MMDDYY, etc.

    Partial formats:  
    - MON YYYY, YY MON, MON YY

    ---
    🔍 Inference Logic:

    - Use fuzzy matching to handle OCR mistakes (e.g., "MGF" → "MFG", "Expry" → "Expiry")
    - If only one date is found and it's associated with "Best Before", "Use by", "Better if used by", "Best if used before", then treat it as the **expiry date**
    - If no labels are present and multiple dates exist:
    - If duration-based statements are present (e.g., “Best before 2 years”), calculate EXP based on MFG
    - Ignore unrelated content (e.g., ingredients, price, nutrition)

     - If multiple dates are present without clear labels, infer:
     - The **earliest date** as the manufacturing date (MFG)
    - The **latest date** as the expiry date (EXP)
    ---
    🧾 Output Format (Always return in this format, and standardize dates if possible):

    mfg date: <mfg_date>  
    expiry date: <expiry_date>

    - If only one is found, return the one in the above format.
    - If neither is found, return:  
      "can't detect the mfg date and expiry date from the Image"
    """
)


human_prompt = HumanMessagePromptTemplate.from_template(
    """
    {text}
    """
)
prompt = ChatPromptTemplate.from_messages([system_prompt, MessagesPlaceholder(variable_name="text"), human_prompt])

# prompt2= ChatPromptTemplate,from_messages 

# Initialize OCR engines
logger.debug("Initializing EasyOCR reader...")
easyocr_reader = easyocr.Reader(['en'])
logger.debug("EasyOCR reader initialized.")

logger.debug("Initializing PaddleOCR reader...")
paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
logger.debug("PaddleOCR reader initialized.")

# Directory to save uploaded files
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.debug(f"Upload directory set to: {UPLOAD_DIR}")

# Mount static files and set up templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
logger.debug("Static files and templates configured.")



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the home page with the upload form.
    """
    logger.debug("Rendering home page.")
    return templates.TemplateResponse("index.html", {"request": request})


    """
    Perform OCR using Tesseract.
    """
    logger.debug(f"Received file for Tesseract: {file.filename}")
    safe_filename = file.filename.replace(" ", "_")
    file_path = UPLOAD_DIR / safe_filename
    logger.debug(f"Saving file to: {file_path}")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.debug("Performing OCR using Tesseract...")
    result = pytesseract.image_to_string(Image.open(file_path))
    logger.debug(f"Tesseract result: {result}")
    return {"tesseract_result": result}


@app.post("/paddleocr/")
async def paddleocr_api(file: UploadFile = File(...)):
    """
    Perform OCR using PaddleOCR.
    """
    logger.debug(f"Received file for PaddleOCR: {file.filename}")
    safe_filename = file.filename.replace(" ", "_")
    file_path = UPLOAD_DIR / safe_filename
    logger.debug(f"Saving file to: {file_path}")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.debug("Performing OCR using PaddleOCR...")
    result = paddleocr_reader.ocr(str(file_path), cls=True)
    paddleocr_text = [line[1][0] for line in result[0]]
    logger.debug(f"PaddleOCR result: {paddleocr_text}")
    return {"paddleocr_result": paddleocr_text}


@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...)):
    """
    Call all three OCR APIs and return the combined results.
    """
    logger.debug(f"Received file for combined OCR processing: {file.filename}")
    safe_filename = file.filename.replace(" ", "_")
    file_path = UPLOAD_DIR / safe_filename
    logger.debug(f"Saving file to: {file_path}")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # PaddleOCR
    logger.debug("Performing OCR using PaddleOCR...")
    paddleocr_result = paddleocr_reader.ocr(str(file_path), cls=True)
    paddleocr_text = [line[1][0] for line in paddleocr_result[0]]
    logger.debug(f"PaddleOCR result: {paddleocr_text}")

    #LLM Classification MFG and Expiry Date
    logger.debug("Performing OCR using LLM...")
    llm_result = model.invoke(prompt.format(text=paddleocr_text))
    llm_text = llm_result.content
    logger.debug(f"LLM result: {llm_text}")   



    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "image_url": f"/static/uploads/{safe_filename}",
            "paddleocr_result": paddleocr_text,
            "extracted_date": llm_text,
        },
    )


if __name__ == "__main__":
    logger.debug("Starting FastAPI application...")
    uvicorn.run(app, host="127.0.0.1", port=8080)