## story_00914561
* greet
 - utter_ask_howcanhelp
* inform{"cuisine": "italian"}
 - utter_on_it
 - utter_ask_location
* inform{"location": "paris"}
 - utter_ask_numpeople
* inform{"people": "six"}
 - utter_ask_price
* inform{"price": "cheap"}
 - utter_ask_moreupdates
* inform{"people": "two"}
 - utter_ask_moreupdates
* inform{"location": "madrid"}
 - utter_ask_moreupdates
* inform{"cuisine": "spanish"}
 - utter_ask_moreupdates
* deny
 - utter_ack_dosearch
 - action_search_restaurants
 - action_suggest
* deny
 - utter_ack_findalternatives
 - action_suggest
* deny
 - utter_ack_findalternatives
 - action_suggest
* affirm
 - utter_ack_makereservation
* thankyou
 - utter_goodbye

