# Class for data collection - create a csv file of phishing and legitimate mails

import base64
import os
import re
import time
import spacy
import pandas as pd
import bs4
import email
import Utils
import win32com.client


# Reads the mail from "prachi.patil@mycit.ie"
def read_outlook_mail():
    # Loads the outlook application
    outlook = win32com.client.Dispatch('outlook.application').Application.GetNamespace("MAPI")
    # Selects the account with the email id passed
    account = outlook.Folders.Item("prachi.patil@mycit.ie")
    # Selects the Inbox folder of the account selected
    acc_inbox = account.Folders.Item("Inbox")
    # Loads all the mails from the Inbox folder
    ham_mails = acc_inbox.Items
    count = 0
    mail_list = []
    # Iterate over the list of mail taken from the Inbox
    for mail in ham_mails:
        if count > 450:
            break
        # Textual content from the email body
        body = mail.Body
        file_name = "ham_mail_" + str(count)

        # Reference: "https://docs.microsoft.com/en-us/office/vba/api/outlook.propertyaccessor#example"
        # Headers of the mail
        header = mail.PropertyAccessor.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x007D001F")
        # Calls the get headers function which separates the headers
        header_dict = get_headers(file_name,header)
        # creates a dictionary of file_name, body and mail item
        mail_row = {"file_name": file_name,
                    # "header": header,
                    "body": body,
                    "mail": mail,
                    "phish": 0}
        count = count + 1
        # Updates the dictionary with separated headers
        mail_row.update(header_dict)
        # Adds the dictionary to the mail list
        mail_list.append(mail_row)
    return mail_list


# Removes the <> punctuation from return path header value
def clean_return_val(header_row):
    if "return-path: " in header_row.keys():
        return_path = header_row["return-path: "]
        if return_path is not None:
            if return_path.strip().endswith(">"):
                return_path = return_path[:-1]
            if return_path.strip().startswith("<"):
                return_path = return_path[1:]
    else:
        return_path = "na"
    header_row["return-path: "] = return_path
    return header_row


# Gets display name from the "From" header
def from_header(header_row):
    email = ""
    from_val = header_row.get("from: ")
    if isinstance(from_val, float) or (from_val is None):
        username = "na"
        email = "na"
    else:
        from_val = from_val.strip()
        # print(from_val)
        if from_val.__contains__("To"):
            res = from_val[:from_val.index("To")]
        else:
            res = from_val
        # Some mail have from header with improper formatting so instead of processing the headers,
        # "na" Not available value is given to those mails.
        # One such example encountered was from header with only < not bracket was not complete.
        # Such header are purposely crafted to create exception.
        if res.__contains__("<") and not res.__contains__(">"):
            email = re.search(r"\<([a-zA-Z0-9._\-\"]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", res).groups()[0]
            username = res[:res.index("<")].strip()
        else:
            # regex to extract email address from from header
            # from format has mail address in <>
            # in case of only mail id  no <> brackets
            if res.__contains__("<"):
                email = res[res.index("<") + 1:-1]
                username = res[:res.rfind("<")]
            else:
                email = "na"
                username = res
        # Some username have " " punctuation marks in beginning and ending of the username, so startswith and endswith
        # function are used to remove only those punctuations and not the one present as a part of username
        if username.strip().startswith("\""):
            username = username[1:]
        if username.strip().endswith("\""):
            username = username.strip()[:-1]
        if email == "":
            email = "na"
        else:
            if email.strip().startswith("<"):
                email = email[1:]
            if email.strip().endswith(">"):
                email = email.strip()[:-1]
    header_row["username"] = username
    header_row["from: "] = email
    return header_row


# Separates headers using regex
def get_headers(file_name, text):
    headers = re.split(r"(^[a-zA-Z-]+: )", text, flags=re.MULTILINE)

    # first entry in header list is from which does not have colon to separate header from header value.
    # So first entry is processed manually
    [headers.remove(word) for word in headers if
     (word == "" or word.startswith("From jose@monkey.org ") or word.startswith("From MAILER-DAEMON"))]

    header_dict = {"file_name": file_name}
    print("Processing mail : ", file_name)

    # The list has one entry of header and next entry of the first header's value,
    # and continues in the same pattern throughout the list
    for i in range(0, len(headers), 2):  # increment by 2,
        header = headers[i].lower()
        # Ignoring all X- headers as they are not useful for header analysis
        if not (header.startswith("x") or header.startswith("X") or header.startswith("arc")):
            value = headers[i + 1].strip()
            if header in header_dict.keys():
                prev_value = header_dict.get(header)
                # Using &&&& as delimiter for distinguishing multiple values with same header (eg: received field)
                value = prev_value + "&&&&" + value
            # Create list of unique headers to be added in dataframe
            header_dict[header] = value
    header_dict = from_header(header_dict)
    header_dict = clean_return_val(header_dict)
    return header_dict


# Removes HTML tags from the etxt
def get_content(message_body):
    # Parses HTML content found in the text
    html_body = bs4.BeautifulSoup(message_body, 'html.parser')
    # Finds any script or style tags in string
    for tag in html_body(["script", "style"]):
        # Removes the tag
        tag.decompose()
    # Get strings from html parsed
    strips = list(html_body.stripped_strings)
    # Removes white space character from the string
    body = [line.replace("\r", "").replace('\n', '').replace("\xa0", ' ').replace('=', '').replace("&nbsp", "") for line
            in strips]
    # returns clean body
    return body


# Gets payload from email with multipart content type
def nested_payload(payload_list):
    text = ""
    # Checks payload type
    if isinstance(payload_list, str):
        text = text + payload_list
    else:
        for payload in payload_list:
            # For emails with text content
            if isinstance(payload, email.message.Message) and payload.get_content_type().__contains__('text'):
                # Checks for base64 encoded strings in mail body
                if str(payload.get('content-transfer-encoding', '')).lower() == 'base64':
                    message_body_content = base64.b64decode(payload.get_payload(), altchars=None).decode('utf-8')
                    text = text + message_body_content
                else:
                    # If mail body is not encoded then gets the payload
                    text = text + payload.get_payload()
            elif isinstance(payload, email.message.Message) and payload.get_content_type().__contains__('multipart'):
                # For emails with multipart content type
                text = nested_payload(payload.get_payload())
    return text


# Removes stopwords, punctuation and lemmatise the text passed
def cleaning_txt(text):
    eng_dict = spacy.load("en_core_web_sm")
    all_stopwords = eng_dict.Defaults.stop_words

    text = re.sub(r'<.*?>', '', text)  # remove all <>
    text = re.sub(r"([a-zA-Z]+)(\d+)", r"\1", text)
    # () is capturing group
    # [a-zA-Z] selects the letter, + selects multiple letters,
    # \d selects digits, d+ selects multiple digits
    # r"\1" selects only first group
    # Above regex replaces "have20" with only "have", removing digit 20.
    punctuations = r'!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
    # Each character in punctuations, is set as a key in translate_dict with " "(space as a value)
    translate_dict = dict((c, " ") for c in punctuations)

    # maketrans method replaces the punctuation sign by its ascii value
    # maketrans(str1,str2) --> str1: characters to replace ; str2: replacement character
    translate_map = str.maketrans(translate_dict)

    # translate method replaces the keys of translate map found in the text, with corresponding value of the key
    text = text.translate(translate_map)

    # Removing stop words
    tokens = [word for word in text.split() if word.lower() not in all_stopwords]
    text = ' '.join(tokens)


    # To lemmatize each word, the word should be passed to an english dictionary, to get its information
    text = " ".join(word.lemma_ for word in eng_dict(text))
    return text


# Returns base64 decoded string
def decode_base64(text):
    return base64.b64decode(text, altchars=None).decode('utf-8')


# Get email payload from the email.message.Message class
def get_email_payload(email):
    body_str = ""
    # lang = ""
    encoding = email.get_content_type()
    try:
        if encoding.__contains__('multipart'):
            text = nested_payload(email._payload)
        else:
            content_encoding = [header[1] for header in email._headers if header[0].__contains__("Content-Transfer-Encoding")]
            if len(content_encoding) > 0 and content_encoding[0] == "base64":
                text = decode_base64(email.get_payload())
            else:
                text = email.get_payload()
        if type(text) is list:
            for message_body in text:
                body_text = get_content(message_body)
        else:
            body_text = get_content(text)
        for body_list in body_text:
            body_list = body_list.strip()
            body_str = body_str + body_list + " "
        body_str = cleaning_txt(body_str)
    except Exception as e:
        e
    return body_str


# Create eml file from the string
def create_eml(email_str):
    return email.message_from_string(email_str)


# Separate mail header and body
def separate_body_header(email_str):
    # seperator_pattern is used to separate body and header in the mail.
    # In the mails taken from the dataset, "X-UID" is last mail header, after which the mail body starts
    seperator_pattern = r"X-UID: \d+"
    # In some mails of the dataset, the last mail header is of "Status"
    # and not X-UID, so different delimiter is used for such cases
    if not email_str.__contains__("X-UID:"):
        seperator_pattern = r"Status: (?:O|RO)"
    # E-Mail is divided into header and body using the seperator pattern with regex
    mail_parts = re.split(seperator_pattern, email_str)
    # The fist part of the mails_part contains mail header
    headers = mail_parts[0]

    # email_body = mail_parts[1]

    # The mail string is converted to eml for easier cleaning of the email body
    email_eml = create_eml(email_str)
    # The eml string is then passed to clean_email_body() function for removing hyperlinks and
    # HTMl tags and retaining only useful string
    email_body = get_email_payload(email_eml)

    return headers, email_body


# Reads the phishing mails text from raw_data directory and creates a list of phishing mail with header and body
def create_phish_mails_list():
    # line to separate mail from each other, only for phishing mails
    seperator_text = 'From jose@monkey.org'

    # Initialising a string to store separated mails
    mail = ""

    # for naming the files
    count = 1

    # Variable used for creating dataframe of raw phishing mails
    mail_list = []

    # path to phishing data taken from jose
    phish_data_path = "../data/raw_data/"

    # Iterating over files from the year 2018 - 2020
    for file in os.listdir(phish_data_path):
        # open the txt file containing combined phishing mails
        with open(phish_data_path + file, 'r', encoding="utf-8") as phish_mails:
            # Read one line at a time from the file
            for line in phish_mails:
                # Only if the line is not seperator line then the line is
                # appended to the mail string
                if seperator_text not in line:
                    # separating a mail from txt file
                    mail = mail + line
                else:
                    if count == 316:
                        search=True
                    file_name = "phish_mail_" + str(count)
                    count += 1
                    # pass the mail contents to separate_body_header function
                    header, body= separate_body_header(mail)
                    header_dict = get_headers(file_name=file_name,text=header)

                    mail_row = {"file_name": file_name,
                                "body": body,
                                "mail": mail,
                                "phish": 1}
                    mail_row.update(header_dict)
                    mail_list.append(mail_row)
                    # seperator file is found indicating that the mail is completed and
                    # new mail is starting from this point forward
                    mail = ""
    return mail_list


# Main function for initial setup of the project - Data collection
def main():
    start_time = time.time()
    print("Start: ", start_time)
    # CSV file containing all the mails
    all_mails_csv = "../data/all_mails_new.csv"

    # Create CSV file for storing raw emails
    phish_mails_list = create_phish_mails_list()
    ham_mails_list = read_outlook_mail()
    mail_list = phish_mails_list + ham_mails_list

    # Create CSV and dataframe of all the mails
    mails_df = pd.DataFrame(mail_list)
    Utils.create_csv(mails_df, all_mails_csv)

    print("done")
    end_time = time.time()
    print("end time: ", end_time)
    print("Total time: ", (end_time - start_time)/60)

if __name__ == "__main__":
    main()
