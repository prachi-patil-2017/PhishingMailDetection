import pathlib
import time
from collections import Counter
import re
import dkim
import dns
import spf
import pandas as pd
import Utils


# Gets DKIM result
def dkim_result(mail_content):
    result = False
    # Result is set to false for no mail body
    if isinstance(mail_content, float):
        result = False
    else:
        # Legitimate mails are stored as outlook objects which are not formmated properly and hence are not processed correctly.
        # Such mails are not processed.
        if not mail_content.__contains__("win32com.gen_py.Microsoft Outlook"):
            try:
                # Performs DKIM check on the mail
                # Result is True is signture is verified else False
                result = dkim.verify(mail_content.encode('utf-8'))
            except dns.resolver.NoNameservers:
                result = False
            except Exception:
                result = False
    return 1 if result else 0


# Gets SPF result
def spf_result(received_val, from_val):
    result = 0
    last = True
    # Checks for floating values
    if not isinstance(received_val, float):
        # Also check for localhost ip address eg:header_2020_126.txt
        received_val_list = received_val.split("&&&&")
        # Select last valid received header
        if len(received_val_list) != 0:
            while last and len(received_val_list) != 0:
                last_received = received_val_list.pop()
                if re.search(
                        r"[0-9a-fA-F]{0,4}:[0-9a-fA-F]{0,4}:[0-9a-fA-F]{0,4}:"
                        r"[0-9a-fA-F]{0,4}:[0-9a-fA-F]{0,4}:[0-9a-fA-F]{0,4}",
                        last_received) is not None:
                    last = True
                elif last_received.__contains__("Exim"):
                    last = True
                elif not last_received.startswith("by"):
                    last = False
                else:
                    last = True

            # Selecting only mail address and ipaddress from the last received header.
            # Using "by" as delimiter only the required part is selected
            # last_received = last_received[:last_received.index(")")]
            if last_received.__contains__("by"):
                last_received = last_received[:last_received.index("by")]

            # Regex search is for complete string and match checks only in the beginning of the string.
            ip_addr_match = re.search(r"(\d{2,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", last_received)
            if ip_addr_match is not None:
                ip_addr = ip_addr_match.group(1)
                # Using group(1) of regex to select 1st capturing group of the matching pattern
                mail_server_match = re.search(r"([a-zA-Z0-9._\-\"]+\.[a-zA-Z0-9_-]+)",
                                              last_received.replace(ip_addr, ""))
                if mail_server_match is not None:
                    mail_server = mail_server_match.group(1)
                else:
                    mail_server = ""

                try:
                    # Gets SPF result
                    result = spf.check(i=ip_addr, s=from_val, h=mail_server)
                    if result.__contains__("pass"):
                        result = 1
                    elif result.__contains__("neutral"):
                        result = 0
                    else:
                        result = -1
                except AttributeError:
                    result = 0
    return result


# Creates DKIM and SPF header result
def auth_result_feature(auth_val, mail, received, from_val):
    auth_res_dict = {"DKIM": 0, "SPF": 0}
    if from_val == "na":
        return auth_res_dict
    # Checks for floating values
    if not isinstance(auth_val, float):
        # Authentication result contain result of mail authentication system separated by ;
        # The value of header is passed to list by splitting
        auth_res_list = auth_val.split(";")
        # Iterate over the list
        for res in auth_res_list:
            # The result is in format -> authentication method 1 = result_1; authentication method 2 = result_2;
            if res.__contains__("="):
                # The result is split into 2;
                pair = res.strip().split("=", 1)
                # The first section has authentication method
                key = pair[0].lstrip()
                # The second section has result
                value = pair[1]
                # result stored in key value pair
                # Only DKIM and SPF methods are selected.
                if key in auth_res_dict.keys():
                    # Result is 1 only if the SPF check passes
                    if value.startswith("pass"):
                        auth_res_dict.update({key: 1})
                    # Result is -1  if the SPF check fails
                    elif value.startswith("fail"):
                        auth_res_dict.update({key: -1})
                    # Result is 0 if the SPF check is none
                    elif value.startswith("none"):
                        auth_res_dict.update({key: 0})
    else:
        # If authentication result is not present, manually verify the result
        # Manual verification of DKIM signature
        dkim_res = dkim_result(mail)
        if not isinstance(received, float):
            # Manual verification of SPF record
            spf_res = spf_result(received, from_val)
            auth_res_dict = {"DKIM": dkim_res, "SPF": spf_res}
        else:
            auth_res_dict = {"DKIM": 0, "SPF": 0}
    # print(auth_res_dict)
    return auth_res_dict


# Calculate similarity between display name and username of the from header
def username_email_match(from_val, display_name):
    result = 0
    # If the from or username value is float or na, then result is set to 0
    if isinstance(from_val, float) or isinstance(display_name, float):
        return 0
    from_val = from_val.strip().lower()
    display_name = display_name.strip().lower()
    if from_val == "na" or isinstance(from_val, float):
        result = 0
    elif from_val == "na" and display_name == "na":
        result = 0
    elif from_val == display_name:
        result = 1
    # From value is splitted in 2 parts: username and domain
    elif from_val.__contains__("@"):
        # example: "Ignacio.Castineiras@cit.ie" username = Ignacio.Castineiras; domain = cit.ie
        user_name = from_val.split("@", 1)[0]
        # Email id with . for example: "Ignacio.Castineiras@cit.ie"
        if user_name.__contains__("."):
            user_name_parts = user_name.split(".")
            if display_name.__contains__(" "):
                display_name_parts = display_name.split(" ")
                # Common values between the username and display name
                match_set = set(user_name_parts) & set(display_name_parts)
                # Calculate total parts int username and display name
                total_len = len(display_name_parts) + len(user_name_parts)
                match_len = len(match_set)
                result = match_len / total_len
        # Email id without space such as "studentservices@mycit.ie"
        else:
            # Prachi Patil
            if display_name.__contains__(" "):
                count = 0
                display_name_parts = display_name.split(" ")
                common = [count + 1 for part in display_name_parts if user_name.__contains__(part)]
                count = len(common)
                result = count / len(display_name_parts)
            # Checks for matching username and display name
            elif user_name.__contains__(display_name):
                result = 1
    return result


# Calculates the number of relay servers
def received_server_number(received):
    if isinstance(received, float):
        return 0
    else:
        # The value of received header is concatenated by &&&&
        return len(received.split("&&&&"))


# return path and from header check
def return_path_from_check(from_val, return_val):
    # The from and return path is checked for floating values or na
    if from_val == "na" or return_val == "na" or from_val is None or return_val is None or isinstance(return_val,float):
        return 0
    from_val = from_val.strip().lower()
    return_path = return_val.strip().lower()
    if return_path.__contains__(";"):
        return_path = list(dict.fromkeys(return_path.split(";")))[0].strip().lower()
    # email id are case insensitive
    if from_val == return_path:
        return 1
    else:
        return 0


# Count the number of subdomains in the from email
def count_subdomain(from_val):
    subdomain_count = 0
    #  format email_username@domain
    if not isinstance(from_val, float) and from_val.__contains__("@") :
        domain = from_val[from_val.index("@") + 1:]
        subdomain_count = Counter(domain)["."] + 1
    # Count the number of subdomain by counting the number "." in the domain
    return subdomain_count
# Prachi Patil


def create_headers_feature_df(header_df):
    header_features_list = []
    # Iterates over each email in the dataframe
    for index, entry in header_df.iterrows():
        file_name = entry["file_name"]
        print(file_name)
        if file_name == "ham_mail_1":
            search = True
        auth_result = auth_result_feature(entry["authentication-results: "], entry["mail"],
                                          entry["received: "], entry["from: "])
        subject = Utils.check_threatening_words(entry["subject: "])
        received_server_no = received_server_number(entry["received: "])
        subdomain_count = count_subdomain(entry["from: "])
        username_emailuser_match = username_email_match(entry["from: "], entry["username"])
        same_return_from = return_path_from_check(entry["from: "], entry["return-path: "])

        header_feature_row = {"catchy_subject": subject,
                              "received_server_number": received_server_no,
                              "subdomain": subdomain_count,
                              "username_email_similarity": username_emailuser_match,
                              "same_return_from": same_return_from,
                              "phish": entry['phish']}
        header_feature_row.update(auth_result)

        header_features_list.append(header_feature_row)
    mails_pd = pd.DataFrame(header_features_list)
    return mails_pd


def header_analysis():
    print("***********************          Performing header Analysis         ***********************")
    start_time = time.time()
    print("header Start: ", start_time)
    path = "../data/header_features.csv"
    file = pathlib.Path(path)
    # Check for header feature file, if file is not present then it creates the file
    if not file.exists():
        # Path to file with email headers
        path = "../data/all_mails.csv"

        # Create features from email headers
        header_features = create_headers_feature_df(pd.read_csv(path))
        # Store the features file for further processing
        Utils.create_csv(header_features, "../data/header_features.csv")
    else:
        header_features = pd.read_csv(path)

    # Train, test and measure the performance of the models
    Utils.result_list = []
    results = Utils.predict_results(header_features, type="header")
    Utils.create_df_csv(results, "../data/header_results.csv")
    end_time = time.time()
    print("header end time: ", end_time)
    print("Total time for header processing in minutes : ", (end_time - start_time)/60)


if __name__ == "__main__":
    header_analysis()
