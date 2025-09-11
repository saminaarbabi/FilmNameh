import asyncio
import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# --- Pydantic Data Models ---
class UserQuery(BaseModel):
    prompt: str = Field(..., min_length=1)

# --- Prompt Engineering Logic ---
def engineer_screenwriting_prompt(user_query: str) -> str:
    """
    این تابع ابتدا قصد کاربر را تشخیص می‌دهد و سپس پرامپت مناسب را تولید می‌کند.
    """
    ADVICE_KEYWORDS = [
        'چطور', 'چگونه', 'چطوری', 'راهنمایی', 'تکنیک', 'روش', 'ایده برای', 'توضیح بده', 'چجوری'
    ]

    is_advice_request = any(keyword in user_query.lower() for keyword in ADVICE_KEYWORDS)

    if is_advice_request:
        prompt_template = f"""
شما یک مشاور و مدرس حرفه‌ای فیلمنامه‌نویسی با درک عمیق از فرهنگ و هنر ایران هستید. وظیفه شما پاسخ دادن به سوالات کاربران در مورد تکنیک‌های داستان‌پردازی و مفاهیم خلاقانه است.

**نکات مهمی که باید رعایت کنید:**
- **زبان خروجی:** تمام پاسخ شما باید به زبان فارسی باشد.
- **ارتباط فرهنگی مثال‌ها:** اگر سوال کاربر به موضوعات معنوی، مذهبی یا فرهنگی (مانند مثال‌های مرتبط با فرهنگ ایران و اسلام) اشاره دارد، مثال‌های شما نیز باید از همین بستر فرهنگی و مرتبط باشند (مثلاً فیلم‌های سینمای ایران یا داستان‌های مذهبی). در غیر این صورت از مثال‌های عمومی سینمای جهان استفاده کن. 
- **پاسخ کاربردی:** پاسخ‌های شما باید واضح، عملی و قابل اجرا باشند.
- **ساختار منظم:** پاسخ خود را به صورت ساختاریافته ارائه دهید.
- **لحن دوستانه:** با لحنی راهنما و دوستانه پاسخ دهید.

**سوال کاربر:**
"{user_query}"

حالا به عنوان یک مشاور حرفه‌ای، به این سوال پاسخ دهید:
"""
    else:
        prompt_template = f"""
شما یک دستیار هوشمند فیلمنامه‌نویسی حرفه‌ای هستید. وظیفه شما این است که بر اساس درخواست کاربر، یک صحنه فیلمنامه با فرمت استاندارد جهانی تولید کنید.

**نکات مهمی که باید رعایت کنید:**
- **زبان:** کل خروجی و کلمات باید به زبان فارسی باشد.
- **فرمت:** از فرمت استاندارد فیلمنامه‌نویسی استفاده کنید و دقت کن متن فارسی باشد (سرصحنه، توضیحات صحنه، نام شخصیت، دیالوگ).
- **خلاقیت:** صحنه را با جزئیات جذاب، تنش و دیالوگ‌های قوی بنویسید.
- **تمرکز:** فقط و فقط صحنه درخواست شده را بنویسید و از توضیحات اضافی خارج از فرمت فیلمنامه خودداری کنید.

**درخواست کاربر:**
"{user_query}"

حالا صحنه را بنویس:
"""
    return prompt_template.strip()

# --- FastAPI App Setup ---
app = FastAPI(title="دستیار هوشمند فیلمنامه‌نویسی", version="4.0.0")

# Configure CORS to allow communication with your local frontend
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "null"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoint ---
@app.post("/generate-stream/")
async def generate_stream_endpoint(query: UserQuery):
    try:
        engineered_prompt = engineer_screenwriting_prompt(query.prompt)
        
        async def stream_generator():
            # A generator function that yields content from the Ollama stream
            stream = ollama.chat(
                model='qwen:7b-chat', # <-- The model has been changed here
                messages=[{'role': 'user', 'content': engineered_prompt}],
                stream=True
            )
            for chunk in stream:
                content = chunk['message']['content']
                yield content
                await asyncio.sleep(0.01)

        # Return the StreamingResponse
        return StreamingResponse(stream_generator(), media_type="text/plain; charset=utf-8")

    except Exception as e:
        # Raise an HTTPException with a more descriptive error message
        raise HTTPException(status_code=500, detail=f"خطایی در پردازش درخواست رخ داد: {str(e)}")

# --- Uvicorn Server Launch ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)