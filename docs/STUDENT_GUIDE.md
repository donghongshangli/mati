# Mati Student Guide

Welcome to the Mati Learning Companion! This guide will help you get started with the application and make the most of its features.

## 1. What is Mati?

Mati is an offline AI learning companion for Grade 10 students in Nepal. It helps you study Computer Science, Science, and English using interactive content and AI assistance, even without internet access.

## 2. Installation

Your teacher or system administrator will likely install Mati for you. If you need to install it yourself, please refer to the detailed [Installation Guide](https://github.com/aa-sikkkk/mati/blob/master/readme.md#installation).

## 3. Starting the Application

Mati is a **graphical application** — there is no terminal/command-line mode.

### Method 1: Using the Application (Recommended)

1. Locate the **`MatiGUI.exe`** file (usually provided by your teacher or in the `dist` folder).
2. Double-click the file to launch the application.
3. The graphical interface will open, ready for use.

### Method 2: Running from Source (Advanced)

If you are using the source code directly, open your terminal or command prompt in the `Mati` directory **only to start the app**, then run:

```bash
python -m student_app.gui_app.main_window
```

Everything after that happens inside the window — you never need to type commands into the terminal.

The application will load and present you with the welcome screen, where you enter your username to begin.

## 4. Navigating Content

The content is organized by Subject, Topic, Subtopic, and Concept.

- Use the sidebar and on-screen buttons to navigate through the different levels.
- You will be able to select a grade, then a subject, then a topic, then a subtopic, and finally a concept.
- Each concept will have explanations and associated questions.

## 5. Learning with Questions and AI Assistance

Once you are viewing a concept, you can interact with the questions related to it.

- **Answering Questions**: Type your answer into the answer box and submit it.
- **Feedback**: The AI grades your answer and explains what was right or missing.
- **Viewing Explanations**: Every graded answer comes with a short explanation of the correct concept.
- **Asking the AI (Q&A)**: Use the **❓ 提问** page to ask the AI questions about the loaded content. The AI will use its knowledge to provide relevant answers.

Remember, all of this works offline!

## 6. Tracking Your Progress

Mati tracks your learning progress locally on your device using your username.

- The application records your answers and whether they were correct.
- You can access progress-related features from the sidebar:
    - **📊 学习进度**: See a summary of your performance and areas for improvement.
    - **⚙️ 进度管理**: Export, import, or reset your progress data. Imported files are validated before they are accepted.

This data helps you and your teacher understand which concepts you have mastered and which areas might need more focus.

## 7. Offline Functionality

One of the key features of Mati is that it works entirely offline. Once installed, you do not need an internet connection to access the content or use the AI assistance.

## 8. Troubleshooting

If you encounter any issues while using the application, inform your teacher or the system administrator. They can refer to the [Technical Implementation Guide](docs/TECHNICAL_IMPLEMENTATION.md#Troubleshooting) or the project documentation for solutions. 
