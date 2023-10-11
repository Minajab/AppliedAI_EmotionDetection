# AppliedAI_EmotionDetection
Applied AI course project in Concordia University

Welcome to A.I.ducation Analytics, where the future of AI-driven academic feedback takes shape. Dive into a world where, as AI lectures unfold complex algorithms, instructors are no longer in the dark! Your innovative system scrutinizes students’ facial responses in real time, distinguishing the curious from the overwhelmed. A sleek dashboard offers the instructor immediate insights—30% engaged, 20% neutral, and 10% nearing cognitive overload. Smart AI suggestions nudge adjustments in real-time, ensuring lectures evolve in response to learners’ needs. As graduate students on this frontier project, you’re not just coding—you’re sculpting the next phase of dynamic, AI-enhanced education.

## Project Objective

The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities. Your system has to be able to analyze images in order to recognize four classes:

1. Neutral: A student presenting neither active engagement nor disengagement, with relaxed facial features.
2. Engaged/Focused: A student evidencing signs of active concentration, with sharp and attentive eyes.
3. Bored/Tired: A student displaying signs of weariness or a lack of interest. This can be evidenced by droopy eyes or vacant stares.
4. Angry/Irritated: Signs of agitation or displeasure, which might manifest as tightened facial muscles, a tight-lipped frown, or narrowed eyes.

The whole project is split across three parts, which build on each other to deliver the final, complete project.

## Project Parts

### Part 1: Data Collection, Cleaning, Labeling & Preliminary Analysis 

In this part, you will develop suitable datasets that you will later need for training your system. This includes examining existing datasets, mapping them to your classes, performing some pre-processing, and basic analysis. More details about this part will be provided below.

### Part 2: CNN Model Development and Basic Evaluation

After learning about CNNs and PyTorch, you will build basic CNN models suitable to detect the above classes. This includes deciding on the number of layers, the type of layers (convolutional, pooling, fully connected), and activation functions. You will have to apply basic evaluation metrics such as accuracy, confusion matrix, precision, recall, and F1-score.

### Part 3: Bias Analysis, Model Refinement & Deep Evaluation

Addressing bias and refining models is advanced and crucial for AI’s ethical applications. You will use visualization tools and metrics to identify bias in your model towards particular classes. Here, you will have to consider demographic fairness to check for racial, age, or gender biases. You will then apply bias mitigation techniques to reduce bias, which can involve oversampling underrepresented classes, employing fairness constraints, or using techniques like adversarial debiasing. Additionally, you have to apply deeper evaluation techniques, in particular, k-fold cross-validation.

