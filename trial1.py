import requests
import json

body = {
    "text_corpus": "If you have always wanted to learn Spanish, you're nowhere near alone; tens of millions of people study the language every day around the world. And it's no wonder! Spanish is a language full of beauty, layers of meaning, and a rich linguistic history. It can take you all over the world, from South America to Spain and even to the islands of the Pacific. But you might have a lot of questions about what it takes to get started - or why it's even worthwhile at all. The good news is you can rest assured that learning the Spanish language is an effort worth undertaking. With the right technology to guide you in your journey, you'll see your efforts pay off in so many ways.",
    "assessment": {
        "name": "Learn Spanish 204",
        "course_id": 21,
        "number_of_questions": 1,
        "assessment_type": "MCQ",
        "is_active": True
    }
}

reply = requests.post("https://curriculai.hertzai.com/quesAnws", data=json.dumps(body), headers={'Content-Type': 'application/json'})
print(json.loads(reply.content))
