# RAGOps Explained Like You're 5

## What is this thing?

Imagine a doctor has a **really, really big bookshelf** full of medical books — so big that no human could ever read all of them. This project builds a **robot helper** that reads the whole bookshelf for the doctor, and when the doctor asks a question, the robot quickly finds the right pages and gives an answer.

That's it. That's the whole project.

---

## The four helpers inside

Our robot is actually four little helpers working together, like a team. Each one lives in its own little house (we call those "containers"), and they all talk to each other on a tiny phone line.

### 🧠 Helper 1 — The Librarian (ChromaDB)

Think of all the medical pages cut into tiny little sticky notes. The Librarian writes a secret number on every sticky note that describes what the note is about — like "this one is about hearts" or "this one is about blood pressure". When you ask a question, the Librarian uses the secret numbers to find the sticky notes most like your question, **super fast**.

The Librarian lives at door number `8000`.

### 🤖 Helper 2 — The Talker (Ollama + LLaMA)

Once the Librarian hands over the right sticky notes, the Talker reads them and writes a friendly answer in normal-people words. The Talker is the "smart robot" part — it's actually a big AI brain called **LLaMA** that lives inside a video-game graphics card so it can think really fast.

The Talker lives at door number `11434`.

### 📋 Helper 3 — The Notebook Keeper (MLflow)

Every time the robot answers a question, the Notebook Keeper writes down: "On Tuesday at 3pm, the doctor asked about hearts, and our robot took 1.5 seconds to answer, and the answer was good." Later, we can flip back through the notebook to see if our robot is getting smarter or sleepier.

The Notebook Keeper lives at door number `5000`.

### 📞 Helper 4 — The Receptionist (FastAPI)

When the doctor wants to ask something, they don't talk to the helpers directly — they call the Receptionist, who passes the question to the Librarian and then the Talker, and brings the answer back. The Receptionist is the front desk for everything.

The Receptionist lives at door number `8080`.

---

## What happens when you ask a question?

1. You call the Receptionist: *"Hey, what causes heart disease?"*
2. Receptionist asks the Librarian: *"Find me sticky notes about that."*
3. Librarian gives back 5 of the most matching sticky notes.
4. Receptionist hands those notes to the Talker: *"Read these and answer in normal words."*
5. Talker thinks for a second and says: *"Smoking, high blood pressure, eating bad food, not moving around enough."*
6. Receptionist gives that answer back to you.
7. The Notebook Keeper quietly writes the whole thing down.

All of this happens in about **1 to 2 seconds**.

---

## The watchdogs (this is the cool part 🐕)

Robots can get lazy or wrong over time, just like a person who hasn't read a new book in a year. So we have **two watchdogs** that keep an eye on our robot.

### 🐕 Watchdog 1 — The "Are the books still fresh?" dog (PSI Drift)

Every week, this dog peeks at the Librarian's secret numbers. If the new sticky notes look very different from the old ones (because medicine changed), the dog **barks** and says: *"Time to update the bookshelf!"* — and the robot automatically goes online and grabs newer medical articles.

### 🐕 Watchdog 2 — The "Is the robot making stuff up?" dog (XAI / SHAP)

When the Talker writes an answer, this dog checks: *"Did the answer actually come from the sticky notes? Or did the robot just make it up?"* If the robot is making things up (we call this **hallucinating**, like a dream that isn't real), the dog raises a red flag.

This is super important because in medicine, a robot that makes stuff up can be **dangerous**. Real doctors need real answers from real research.

---

## How we make sure the robot stays smart

Every time someone changes the robot's code, a **report card machine** runs:

- ✅ Did the code follow the rules? (lint)
- ✅ Did all the little tests pass? (unit tests)
- ✅ Does the robot still build correctly? (docker build)
- ✅ Did anyone leave a `print()` lying around? (oops-finder)

If any of those fail, the change isn't allowed in. It's like a teacher checking your homework before letting you turn it in.

---

## The folders, in kid-words

```
ragops/
├── serving/        ← The Receptionist's desk
├── rag_pipeline/   ← The Librarian's brain (sticky-note logic)
├── src/            ← The shared toolbox everyone uses
├── mlops/          ← The watchdogs and the Notebook Keeper
├── evaluation/     ← The robot's report card
├── data/           ← The actual medical sticky notes (3,706 of them!)
├── docker/         ← The houses each helper lives in
└── tests/          ← Practice questions to check the robot
```

---

## The really, really short version

> **We built a robot that reads 3,706 medical research papers, answers questions about them in plain English, double-checks its own work so it doesn't lie, and quietly updates itself every week so it never gets out of date.**

That's RAGOps. 🩺🤖
