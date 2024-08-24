import sys
import re
import os
import win32com.client
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import sqlite3
import unicodedata
import json

def remove_diacritics(input_str):
    # Normalize the string to decompose combined characters into their base characters and diacritics
    normalized_str = unicodedata.normalize('NFD', input_str)
    
    # Filter out diacritical marks
    ascii_str = ''.join(c for c in normalized_str if unicodedata.category(c) != 'Mn')
    
    return ascii_str

settings_path = 'settings.json'
settings = {}

def load_emails():
    # Open the file and load the JSON data
    with open(settings_path, 'r') as file:
        settings = json.loads(file.read())

    verbose = settings["verbose"]

    if not os.path.exists(settings["db_path"]):

            # Connect to a database (creates it if it doesn't exist)
            conn = sqlite3.connect(settings["db_path"])
            # Create a cursor object
            cursor = conn.cursor()

            with open(settings["db_sql_schema_path"], 'r') as file:
                schema = file.read()   

            queries = schema.strip().split(";")

            for query in queries:
                if len(query) > 6:
                    query += ";"
                    if verbose: print(query)
                    cursor.execute(query)
            conn.close()

    conn = sqlite3.connect(settings["db_path"])

    # Create a cursor object
    cursor = conn.cursor()

    # Define the save directory
    save_directory = settings["email_path"]
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Connect to Outlook
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

    # Access the inbox
    inbox = outlook.Folders.Item(settings["email_address"]).Folders.Item(settings["email_folder"])

    # Get the items from the inbox
    messages = inbox.Items
    messages.Sort("[ReceivedTime]", True)  # Sort by most recent

    # Collect the last 100 messages manually
    message_list = []
    message_count = 0

    for message in messages:
        message_list.append(message)
        message_count += 1
        if message_count >= settings["email_count"]:
            break

    # Iterate through the messages
    for message in message_list:

        # Insert email to bbdd
        query = f'''
        
            INSERT INTO [EMAIL] (
            [DATE],
            [FROM_NAME],
            [FROM_ADDRESS],
            [TO_NAME],
            [TO_ADDRESS],
            [CC_NAME],
            [CC_ADDRESS],
            [SUBJECT],
            [BODY]
        ) VALUES (
            '{message.SentOn.strftime('%Y-%m-%d %H:%M:%S')}', 
            '{remove_diacritics(message.SenderName.replace("'", "''"))}',
            '{message.SenderEmailAddress.replace("'", "''")}',
            '{", ".join([remove_diacritics(recipient.Name.replace("'", "''")) for recipient in message.Recipients if recipient.Type == 1])}',
            '{", ".join([recipient.Address.replace("'", "''") for recipient in message.Recipients if recipient.Type == 1])}',
            '{", ".join([remove_diacritics(recipient.Name.replace("'", "''")) for recipient in message.Recipients if recipient.Type == 2])}',
            '{", ".join([recipient.Address.replace("'", "''") for recipient in message.Recipients if recipient.Type == 2])}',
            '{remove_diacritics(message.Subject.replace("'", "''"))}',
            '{remove_diacritics(message.Body.replace("'", "''"))}'
        );
        '''
        #if verbose: print(query)
        # Create a table
        cursor.execute(query)
        # Retrieve the last inserted EMAILID
        email_id = cursor.lastrowid

        # Convert message.ReceivedTime to naive datetime
        received_time = message.ReceivedTime

        subject = remove_diacritics(message.Subject or 'No Subject')
        received_time_str = received_time.strftime('%Y-%m-%d_%H-%M-%S')
        email_filename = f"{received_time_str}_{subject}".replace(" ","_")
        email_filepath = os.path.join(save_directory, email_filename)

        # Create a safe filename
        email_folder_name = ''.join(c for c in email_filename if c.isalnum() or c in (' ', '_')).rstrip().replace(" ","_")
        email_folder_path = os.path.join(save_directory, email_folder_name)
        email_filepath = os.path.join(email_folder_path,email_folder_name + ".html")
        
        # Check if the message object has an HTMLBody attribute
        messagebody = message.Body.replace("'", "''")
        if hasattr(message, 'HTMLBody'):
            messagebody = message.HTMLBody        

        body = re.sub(r'\.(png|jpeg|jpg)\@.*?\">', r'.\1'+'"', messagebody).replace('src="cid:','src="images/')
        

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(body, 'html.parser')

    # Ensure the <html> tag exists; if not, create it
        if not soup.html:
            html_tag = soup.new_tag('html')
            soup.insert(0, html_tag)
        else:
            html_tag = soup.html

        # Ensure the <body> tag exists; if not, create it
        if not soup.body:
            body_tag = soup.new_tag('body')
            html_tag.append(body_tag)
        else:
            body_tag = soup.body

        # Create a new div element
        attachments_div = soup.new_tag('div')

        # Add an h1 header with "Attachments"
        h1_tag = soup.new_tag('h1')
        h1_tag.string = "Attachments"
        attachments_div.append(h1_tag)

        # Create a bullet-point list (ul) to contain the attachment links
        ul_tag = soup.new_tag('ul')

        if not os.path.exists(email_folder_path):
                os.makedirs(email_folder_path)
                os.makedirs(os.path.join(email_folder_path, "attachments"))
                os.makedirs(os.path.join(email_folder_path, "images"))
                if message.Attachments.Count > 0:
                    for attachment in message.Attachments:
                        
                        attachment_filename_clean = "images/" + remove_diacritics(attachment.FileName).replace(" ","_")
                        extension = attachment_filename_clean.split('.')[-1]
                        if attachment_filename_clean not in body:
                            attachment_filename_clean = "attachments/" + remove_diacritics(attachment.FileName).replace(" ","_")
                            
    # Create a list item (li) for each attachment
                            li_tag = soup.new_tag('li')
                            # Create a link (a) with the href pointing to the attachment
                            a_tag = soup.new_tag('a', href=attachment_filename_clean)
                            a_tag.string = attachment_filename_clean
                            li_tag.append(a_tag)
                            ul_tag.append(li_tag)

                        attachment_filename = os.path.join(email_folder_path, attachment_filename_clean)

                        # Insert an attachment using the retrieved EMAILID
                        cursor.execute('''
                            INSERT INTO ATTACHMENT (
                                [EMAILID],
                                [NAME],
                                [PATH],
                                [FILE_EXTENSION]
                            ) VALUES (
                                ?,                            -- EMAILID
                                ?,                            -- ATTACHMENT_NAME
                                ?,                            -- ATTACHMENT_PATH
                                ?                             -- ATTACHMENT_TYPE
                            )
                        ''', (email_id, remove_diacritics(attachment.FileName).replace(" ","_"), attachment_filename, extension))

                        if verbose: print(os.path.abspath(attachment_filename))
                        # Save attachment
                        attachment.SaveAsFile(os.path.abspath(attachment_filename))

                        

                # Append the list to the div
                attachments_div.append(ul_tag)

                # Insert the attachments div at the top of the body
                soup.body.insert(0, attachments_div)

                # Get the modified HTML content
                body = str(soup.prettify())

        # Extracting email information
        email_info = {
            "From": message.SenderName,
            "Sent date": message.SentOn.strftime("%B %d, %Y %I:%M %p"),
            "To": ", ".join([recipient.Name for recipient in message.Recipients if recipient.Type == 1]),
            "CC": ", ".join([recipient.Name for recipient in message.Recipients if recipient.Type == 2]),
            "Subject": message.Subject,
            "FromAddress":message.SenderEmailAddress,
            "ToAddress": ", ".join([recipient.Address for recipient in message.Recipients if recipient.Type == 1]),
            "CCAddress": ", ".join([recipient.Address for recipient in message.Recipients if recipient.Type == 2])
        }

        # Create a new div element for the email information
        info_div = soup.new_tag('div')

        # Add each piece of email information to the div
        for key, value in email_info.items():
            p_tag = soup.new_tag('p')
            p_tag.string = f"{key}: {value}"
            info_div.append(p_tag)

        # Insert the information div at the top of the body
        body_tag.insert(0, info_div)

        # Get the modified HTML content
        body = str(soup.prettify())

        # Insert an attachment using the retrieved EMAILID
        cursor.execute('''
            INSERT INTO ATTACHMENT (
                [EMAILID],
                [NAME],
                [PATH],
                [FILE_EXTENSION]
            ) VALUES (
                ?,                            -- EMAILID
                ?,                            -- ATTACHMENT_NAME
                ?,                            -- ATTACHMENT_PATH
                ?                             -- ATTACHMENT_TYPE
            )
        ''', (email_id, 'Message Body', email_filepath, 'html'))

        if verbose: print(email_id)

        # Save email content
        with open(email_filepath, 'w', encoding='utf-8', errors='ignore') as file:
            
            file.write(body)

        # Commit the transaction
        conn.commit()
        if verbose: print(f"Saved: {email_filepath}")

    if verbose: print("Emails retrieval and saving completed.")

    # Get the Calendar folder (9 refers to the Calendar folder)
    calendar_folder = outlook.GetDefaultFolder(9)

    # Get all calendar items and sort them by start time
    items = calendar_folder.Items
    items.Sort("[Start]")  # Sort items by start time
    items.IncludeRecurrences = True  # Include recurring events

    # Convert datetime.now() to aware datetime using the local timezone of the first appointment
    next_appointments = []
    for appointment in items:
        appointment_start = appointment.Start
        if len(next_appointments) >= settings["appointment_count"]:
            break
        
        # Ensure we compare with aware datetime
        now = datetime.now(appointment_start.tzinfo)  # Make now aware using the appointment's timezone
        
        if appointment_start >= now:
            next_appointments.append(appointment)

    # Write the next 10 appointments to a file
    for appointment in next_appointments:
        # Insert email to bbdd
        query = f'''INSERT INTO [APPOINTMENT] (NAME, START_DATE, END_DATE)
                VALUES ('{appointment.Subject}', '{appointment.Start.strftime("%B %d, %Y %I:%M %p")}', '{appointment.End.strftime("%B %d, %Y %I:%M %p")}');
        '''
        if verbose: 
            print(appointment.Subject,query)
        #if verbose: print(query)
        # Create a table
        cursor.execute(query)
        conn.commit()

    if verbose: print("Future appointments saved.")
    # Close the connection
    conn.close()


