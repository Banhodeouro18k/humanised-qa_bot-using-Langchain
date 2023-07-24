# Project Name: Humanized Chatbot with Lang-Chain Framework and OpenAI

Welcome to the Humanized Chatbot project! This chatbot is designed to replace a human presence on messaging platforms like Telegram and WhatsApp. It utilizes the Lang-Chain framework and OpenAI to create a human-like conversational experience for users. The chatbot learns from previous conversations and builds a knowledge graph memory to enhance its responses over time.

## Features

- **Humanized Conversations:** The chatbot is trained to have human-like conversations, making interactions feel natural and engaging.

- **Pattern Recognition:** By analyzing previous conversations, the chatbot identifies patterns and learns from them to improve its responses.

- **Knowledge Graph Memory:** The chatbot maintains a knowledge graph memory that allows it to retain information and provide more contextually relevant answers.

- **Integration with Flask:** The project utilizes Flask, a web framework in Python, for easy integration and deployment of the chatbot.

- **User-friendly Admin Interface:** You can access the chatbot's admin webpage through `/admin` on the Flask server. Here, you can create embeddings without the need for backend development knowledge.

- **Dynamic Embedding:** The latest embedding generated through the admin interface will be automatically incorporated into the chatbot, ensuring it stays up-to-date with the most recent information.

- **Chainlet for Response Integration:** To streamline the response integration process, the project introduces a concept called "chainlet," which shares similarities with a streamlet.

## Getting Started

Follow these steps to set up and run the Humanized Chatbot on your local machine:

### Prerequisites

1. Make sure you have Python installed (version X.X or higher).

2. Install Flask and other required dependencies:
```bash
pip install flask
# Add any other necessary packages here
```

### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your_username/your_project.git
cd your_project
```

2. Run the Flask application:

```bash
python app.py
```

3. Access the chatbot's admin webpage:

Open your web browser and go to `http://localhost:5000/admin`. Here, you can create new embeddings to improve the chatbot's responses.

### Usage

Once the Flask server runs,`app.py` you can create your own embedding, then you can run the `main.py` by running `chainlit run main.py -w`  the chatbot is ready to be used on your desired messaging platform.

### Development

To contribute to the project or make custom modifications, follow these guidelines:

1. Study the Lang-Chain framework and OpenAI documentation to understand the core of the chatbot's functionality.

2. Explore the Flask application in `app.py` to understand how the integration works and how to manage embeddings via the admin webpage.

3. For any additional changes, create a new branch and commit your changes there:

```bash
git checkout -b feature/your_feature_name
```

4. Push the branch to the remote repository:

```bash
git push origin feature/your_feature_name
```

5. Open a pull request and describe the changes you've made. Your contributions are valuable to the project!

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Credits

- [OpenAI](https://openai.com) - For providing powerful language processing capabilities.
- [Flask](https://flask.palletsprojects.com/) - For the web application integration.
- [Lang-Chain Framework](https://python.langchain.com/docs/get_started/introduction.html) - For facilitating pattern recognition and knowledge graph memory.
- [chainlit](https://docs.chainlit.io/overview) - Inspiration for the chainlet concept.

## Future works

- Integration of social media
- More advanced concepts of prompts
- UI/UX for the Application in a more detailed way
- docker 
- CI/CD 

## Contact

If you have any questions or suggestions regarding the project, feel free to contact us at:

- Email: vanamayaswanth@gmail.com

Thank you for using our Humanized Chatbot! We hope it enriches your messaging experience and provides helpful and engaging conversations. Happy chatting!
