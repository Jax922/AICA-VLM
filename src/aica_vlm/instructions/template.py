# Prompt: You are helping build a benchmark for evaluating vision-language models' understanding of emotion.  Task: Classify the facial expression of the person shown in an image. The emotion can be represented in three label sets: 6 basic emotions, 8 emotions, and 26 fine-grained emotions. Please generate 50 different natural instructions that ask the model to classify the emotion shown in an image, varying the phrasing while keeping the meaning the same. The instructions should be clear and specific, and avoid unnecessary complexity. Output a python array list.
EU_FER_instructions = [
    "What facial expression is the person showing in this image?",
    "Identify the facial expression of the person in the picture.",
    "Classify the person’s facial expression in the image.",
    "Describe the facial expression of the individual shown.",
    "What emotion is expressed through the person's face?",
    "Determine the facial expression displayed in the image.",
    "Analyze the person's face and label their expression.",
    "What is the person’s facial emotion in this photo?",
    "Select the appropriate label for the person’s facial expression.",
    "Based on the facial features, what emotion is being shown?",
    "What facial emotion is depicted by the person in this image?",
    "Please identify the emotion shown on the person's face.",
    "Can you describe the facial expression the person has?",
    "Assign an emotion label to the facial expression in the picture.",
    "How would you categorize the facial expression of this individual?",
    "What type of expression does this person display?",
    "Label the facial expression visible in the image.",
    "What feeling is being communicated through the face?",
    "Interpret the facial expression in the image.",
    "Which facial expression is most appropriate for the person shown?",
    "Is the person smiling, frowning, or something else?",
    "Give a name to the facial expression in this photo.",
    "Look at the face and determine the emotional expression.",
    "What best describes the person's facial look in the image?",
    "How would you describe the expression on this person’s face?",
    "Choose the correct label for this facial expression.",
    "What does the person’s face suggest emotionally?",
    "Provide a classification of the facial expression shown.",
    "Identify the mood conveyed by the person’s face.",
    "What expression is the person making with their face?",
    "Name the emotional state reflected in the facial expression.",
    "Categorize the emotion visible in the person's face.",
    "Is this facial expression happy, sad, angry, or something else?",
    "How is the person feeling based on their facial cues?",
    "Determine what the person's face is expressing.",
    "What emotion does this face likely convey?",
    "Please name the facial expression in this image.",
    "Based on facial cues, what is the person feeling?",
    "Pick the correct expression label for the face.",
    "How would you interpret the expression on this person?",
    "Give a descriptive label for the expression shown.",
    "Classify this person’s facial emotion.",
    "What kind of facial expression is this?",
    "What's the emotional expression displayed by the person?",
    "Assess the face and assign an expression label.",
    "Look at the person’s face. What emotion are they expressing?",
    "How would you emotionally interpret the person’s face?",
    "Identify the likely emotion from the person’s facial features.",
    "Can you recognize the facial expression in this image?",
    "Provide your classification of this facial expression.",
]


# Prompt: You are helping build a benchmark for evaluating vision-language models' understanding of emotion.  Task: Classify the emotion of a person shown in an image. The emotion can be represented in three label sets: 6 basic emotions, 8 emotions, and 26 fine-grained emotions. Please generate 50 different natural instructions that ask the model to classify the emotion shown in an image, varying the phrasing while keeping the meaning the same. The instructions should be clear and specific, and avoid unnecessary complexity. Output a python array list.
EU_people_in_wild = [
    "Identify the emotion displayed by the person in the image.",
    "What emotion is the person in the picture showing?",
    "Determine the person’s emotional expression in this photo.",
    "Classify the emotion of the person visible in the image.",
    "Based on the image, what emotion is being expressed by the individual?",
    "Describe the emotion shown by the person in this image.",
    "Looking at this image, what is the person feeling?",
    "Please select the most appropriate emotion label for the person in the image.",
    "From the visual cues, what emotion is the person exhibiting?",
    "Analyze the person’s face and body language and classify the emotion.",
    "What is the primary emotion expressed by the individual in this picture?",
    "Can you tell what emotion this person is experiencing in the photo?",
    "Evaluate the emotional state of the person shown in this image.",
    "Label the emotion that best matches the person’s expression in the image.",
    "How would you describe the person’s emotional state in this picture?",
    "Choose the correct emotion category for the person in the photo.",
    "What feeling is the person in this image expressing?",
    "Given this image, classify the person’s emotion as accurately as possible.",
    "Identify which emotion the person appears to be feeling in this image.",
    "What is the observed emotion of the individual in the image?",
    "Please categorize the emotion conveyed by the person in this visual.",
    "Assign an emotion label to the person in the image.",
    "What emotional expression is visible on the person's face?",
    "Classify the visible emotion of the person based on the image.",
    "Using visual evidence, determine the person’s emotional state.",
    "Select the best fitting emotion for the individual in the image.",
    "What is the most likely emotion depicted by the person?",
    "What emotion is conveyed by the person’s facial expression and body language?",
    "Analyze and name the emotion represented in the image.",
    "Describe the emotion being experienced by the subject in this image.",
    "Look at the image and decide what emotion the person is showing.",
    "What emotion would you assign to this person from the image?",
    "Inspect the photo and indicate the emotion displayed.",
    "Based on facial cues, determine the person’s emotion.",
    "What is the person’s dominant emotion in this picture?",
    "Given the person’s appearance in the image, what emotion is likely?",
    "Which emotion best describes the person’s expression in this image?",
    "Observe the individual in this image and classify their emotion.",
    "From this photo, what emotion can you infer the person is feeling?",
    "Please determine which emotion label applies to this person.",
    "What emotional state does the person appear to be in?",
    "What emotion is reflected in the image of this person?",
    "Evaluate the expression in the image and label the emotion.",
    "Which of the emotion categories fits the person in the photo?",
    "Identify and name the emotion expressed in this visual.",
    "What do you perceive as the emotion of the person in this photo?",
    "Judge the person’s emotion based on their expression and pose.",
    "Classify the emotion according to the facial expression in the image.",
    "What’s the emotional impression conveyed by the person in the image?",
    "What emotion is captured in the image of this individual?",
]


# Prompt: You are helping build a benchmark for evaluating vision-language models' understanding of emotion.  Task: Classify the emotion that the image evokes in a human observer. The emotion can be represented in three label sets: 6 basic emotions, 8 emotions, and 26 fine-grained emotions. Please generate 50 different natural instructions that ask the model to classify the emotion shown in an image, varying the phrasing while keeping the meaning the same. The instructions should be clear and specific, and avoid unnecessary complexity. Output a Python array list.
EU_observer_emotion = [
    "Identify the emotion this image is likely to evoke in a human viewer.",
    "Based on the visual content, what emotion would a typical observer feel?",
    "Determine the emotional response this image is meant to trigger.",
    "Classify the feeling this image might evoke in someone who sees it.",
    "What kind of emotion does this image provoke in a human observer?",
    "Label the emotion this image is most likely to elicit.",
    "From the viewer's perspective, what emotion does this image convey?",
    "Which emotion best represents the reaction a person would have to this image?",
    "Analyze the image and specify the emotion it may evoke in humans.",
    "Identify the most appropriate emotional label based on the image.",
    "Select the emotion this image is intended to trigger in the observer.",
    "What emotional state would this image most likely induce?",
    "Assign an emotional category to this image based on how it may affect viewers.",
    "Classify the image according to the emotion it might generate in people.",
    "Choose the emotion that best describes how this image would make someone feel.",
    "What is the dominant emotion evoked by this image?",
    "Determine the observer’s emotional reaction to this image.",
    "Predict the emotion a viewer might experience from seeing this image.",
    "What emotion is visually communicated through this image?",
    "Identify the emotion that resonates from this image for an average person.",
    "Which emotion category does this image fall under from a viewer’s standpoint?",
    "From an emotional perspective, how should this image be categorized?",
    "Classify the evoked feeling this image might produce.",
    "Name the emotion likely triggered by this image in human viewers.",
    "Label the emotional tone or mood evoked by this image.",
    "Based on human perception, which emotion fits this image best?",
    "Describe the emotional impact this image may have on people.",
    "How would a person emotionally respond to this image?",
    "Select the category of emotion that this image suggests.",
    "Which emotional response is this image most likely to generate?",
    "Choose the emotion that aligns with the feeling this image produces.",
    "Identify which emotion label best fits the viewer’s reaction.",
    "Determine what someone might feel upon seeing this image.",
    "Which emotional state is this image most associated with?",
    "Based on visual cues, what emotion would this image evoke?",
    "How might a human emotionally interpret this image?",
    "Which emotional reaction does this image most strongly suggest?",
    "Classify the emotional effect this image might have on an observer.",
    "What feeling does this image most likely convey to a person?",
    "If a human viewed this image, what would they likely feel?",
    "Assign the appropriate emotional label for this image.",
    "From an affective viewpoint, how does this image register?",
    "Select the primary emotion evoked by the scene shown.",
    "What kind of mood does this image create for the viewer?",
    "Describe the emotional atmosphere this image portrays.",
    "Which emotion is most likely to be felt when viewing this image?",
    "How would this image emotionally affect someone looking at it?",
    "Name the emotional response that best matches this image.",
    "Evaluate the image and choose the emotion it most likely elicits.",
    "Pick the emotion that represents how this image is perceived emotionally.",
    "Based on the content, what emotion would this image trigger in a person?",
]


DES_tail = "Respond with two numerical values: valence and arousal. Valence measures emotional pleasantness, and arousal measures emotional intensity. Each value should be between -1 and 1, rounded to two decimal places. Use the following format: Valence: <value>, Arousal: <value>"

CES_tail = "Please provide the emotion label for the image. The label should be one of the following: "


def build_CES_tail(labels):
    """
    Build the CES tail string with the provided labels.

    Args:
        labels (list): List of emotion labels to include in the tail.

    Returns:
        str: Formatted tail string for CES.
    """
    return f"Please provide the emotion label for the image. The label should be one of the following: {', '.join(labels)}. Using this format: label"


def build_CoT_observer_prompt(emotion_options: list[str]) -> str:
    prompt = f"""
    Instructions:
    You are given an image. Predict the emotion it may evoke in a typical viewer.

    1. Observe: Analyze the image in terms of brightness, colorfulness, scene type, objects, facial expressions, and human actions.

    2. Interpret: Based on these visual features, infer what kind of emotion the image is likely to evoke in an observer.

    3. Select: Choose one emotion from the list that best describes this likely emotional response.

    Emotion Options: {", ".join(emotion_options)}

    Note: Only write the selected emotion on the last line.
    """
    return prompt


def build_CoT_FER_prompt(emotion_options: list[str]) -> str:
    prompt = f"""
    Instructions:
    You are given an image of a person. Identify the emotion expressed through their facial features.

    1. Observe: Describe the person’s facial expression (e.g., eyes, mouth, brows).

    2. Interpret: Infer what the person might be feeling based on their expression.

    3. Select: Choose one emotion from the list that best matches the facial expression.

    Emotion Options: {", ".join(emotion_options)}

    Note: Only write the selected emotion on the last line.
    """
    return prompt


def build_CoT_people_in_wild_prompt(emotion_options: list[str]) -> str:
    prompt = f"""
    Instructions:
    You are given an image of a person. Identify the emotion expressed through their facial features.

    1. Observe: Describe the person's body posture, facial expression, and surrounding environment.

    2. Interpret: Based on the visual context, infer how the person might feel.

    3. Select: Choose one emotion from the list that best fits the person's emotional state.

    Emotion Options: {", ".join(emotion_options)}

    Note: Only write the selected emotion on the last line.
    """
    return prompt


instruction_templates = {
    "EU_FER_instructions": EU_FER_instructions,
    "EU_people_in_wild": EU_people_in_wild,
    "EU_observer_emotion": EU_observer_emotion,
}
