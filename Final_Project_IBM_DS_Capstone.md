>>> ## IBM Applied Data Science Capstone Final Project:
>> # Exploring the Services Available Near and the Quality of Playgrounds in Germany
#### This project is designed to satisfy the IBM Applied Data Science Capstone final project requirements. However, the project is also intended to be useful.

# Table of contents
1. [Introduction](#introduction)
2. [Part I: Data preparation](#part1)
    1. [Part IA: Get a list of playgrounds in a city of interest](#part1A)
    2. [Part IB: Get detailed playground information for each location](#part1B)
    3. [Part IC: Visualizing the playground dataset](#part1C)
    4. [Part ID: Adding Foursquare data](#part1D)
3. [Part II: Methodology and results](#part2)
    1. [Part IIA: Exploring the top five venue types for each playground](#part2A)
    2. [Part IIB: Clustering and mapping based on venues](#part2B)
    3. [Part IIC: Clustering and mapping based on playground equipment](#bonus1)
    4. [Part IID: Finding the playgrounds that are near fast food restaurants, icecream shops, etc.](#bonus2)
4. [Discussion and concluding remarks](#discussionandconclusion)

# Introduction <a name="introduction"></a>

### Description of the problem/background <a name="description"></a>
Before the series of coronavirus lockdowns that we've had in Germany, my spouse and I generally did our shopping and errands while our children were at school. We could then do family trips the nearby playgrounds in the afternoon. Playground trips would occur sometimes several times a week. 
<br /> 
<br />
Now in coronavirus times, things are a bit different. For weeks at a time, our children are home ALL THE TIME in lockdown with us. So, to go do our shopping involves one of us staying with the children while the other is shopping. One option is of course to stay home with them, but they're also quite bored with being in lockdown and some outside time is great for them anyway. So, now that the lockdowns are less intense - playgrounds open but not always schools and daycares - we like to combine shopping and playground trips. This generally involves one of us going into the store while the other takes the children to a playground.
<br />
<br />
To aid the combined shopping-playground process, this project will combine playground and commercial venue data. Using a crowd-sourced playground database and the Foursquare API, we can check which playgrounds are near what sorts of shops. Do we need to go to a variety of stores to meet our shopping requirements? There's a cluster of playgrounds for that. Do we need a set of playgrounds near supermarkets and such? Yep, we can find those. How about playgrounds with extensive equipment or playgrounds that are away from the shops so the children can really go crazy? Yes, we can identify those too. So, there's a couple different problems being discussed here. The primary one is combining shopping and playground trips into the village. The other is identifying playgrounds that fit specific playing needs. That is, playgrounds with a lot of equipment for an extended adventure versus more limited ones for shorter trips. I should be able to address both sorts of information requirements once I collect and prepare the data.
<br />
<br />

### Data plan<a name="data"></a>
The plan is to use html web scraping to retrieve a list of playgrounds and their characteristics from the crowd-source based website 'spielplatznet' (https://spielplatznet.de/spielplaetze). This site allows a user to search for a city in Germany which then returns a list of playgrounds in the vicinity. It is based on playground users inputting the data, so not all areas of the country are well-represented. However, the area where I will conduct the analysis - the village of Wedel in the state of Schleswig-Holstein (near Hamburg) has pretty good data. A plus is that I know many of the locations well and so can confirm when the data is complete or missing.
<br />
<br />
The second substantial data source is the Foursquare API. I will use it to retrieve information on venues near each playground. I can then classify the playgrounds based on what's nearby for the purpose of combining shopping/errands and playground trips to the village. I'll primarily use the Foursquare data in a k-means clustering process, but also to search through for particular types of venues. These will include particular stores, store types, or stores with keywords in their titles such as 'icecream' ('eis' in German).
<br />
<br />
I will also use geolocator to search for the village's geocoordinates. This is probably a little excessive as I could take an average/mean of the playground coordinates.

### Data examples<a name="dataexample"></a>
To aid in planning, I have gathered data from the Spielplatznet and Foursquare websites to make example rows of the dataframes that I will be developing:

####  From the playground information website Spielplatznet:


```python
#The playground data includes identifying information as well as a short description (in German) and I will scrape
#information on the number and types of playground equipment available.
import pandas as pd
pd.set_option('display.max_columns', None)
df_columns=('playground', 'latitude', 'longitude', 'description',
       'rating', 'water feature', 'sandpit', 'cable car', 'playhouse',
       'tree house', 'slide', 'swing', 'climbing features', 'sledding hill',
       'football field', 'seesaw', 'basketball', 'nest swing', 'total equipment')
df=pd.DataFrame([['Spielplatz Waldspielplatz Moorwegsiedlung Wedel',53.5926308917772,9.73169803619385,
    'Großer Spielplatz im Wald. Viel Wiese.',5,0,0,2,0,0,0,1,1,0,0,0,0,0,4]],columns=df_columns)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playground</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>description</th>
      <th>rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>53.592631</td>
      <td>9.731698</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



#### From the Foursquare website:


```python
#This is an example of the data when it has been grouped by playground and mean-normalized for use in 
#the k-means clustering algorithm.
df2_columns=('Playground','Asian Restaurant','Auto Garage','Bakery','Beach','Beach Bar','Boat Rental',
            'Boat or Ferry','Bookstore','Bus Stop','Café','Clothing Store','College Gym','Construction & Landscaping',
            'Drugstore','Electronics Store','Farmers Market','Fast Food Restaurant','Flea Market','Food & Drink Shop',
            'French Restaurant','Furniture / Home Store','Garden','Garden Center','German Restaurant','Gym',
            'Gym / Fitness Center','Harbor / Marina','Hotel','Insurance Office','Italian Restaurant',
            'Light Rail Station','Liquor Store','Mexican Restaurant','Museum','Nightclub','Optical Shop','Pet Store',
            'Pier','Plaza','Pool','Pub','Residential Building (Apartment / Condo)','Restaurant','Sandwich Place',
            'Sculpture Garden','Seafood Restaurant','Shopping Mall','Soccer Field','Spa','Steakhouse','Supermarket',
            'Taverna','Tea Room','Thai Restaurant','Theater','Trail','Trattoria/Osteria','Turkish Restaurant')
df2=pd.DataFrame([['Spielplatz Croningstraße Wedel',0.0625,0.0625,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0.0625,0,0.125,0,0,0.0625,0.0625,0,0,0,0,0.0625,0,0,0,0,0,0,0,0,0.0625,0,
                  0.0625,0,0,0,0,0,0.0625,0.0625,0,0,0,0,0,0,0.187500,0.062500,0,0,0,0,0]],columns=df2_columns)
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>Asian Restaurant</th>
      <th>Auto Garage</th>
      <th>Bakery</th>
      <th>Beach</th>
      <th>Beach Bar</th>
      <th>Boat Rental</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Bus Stop</th>
      <th>Café</th>
      <th>Clothing Store</th>
      <th>College Gym</th>
      <th>Construction &amp; Landscaping</th>
      <th>Drugstore</th>
      <th>Electronics Store</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Flea Market</th>
      <th>Food &amp; Drink Shop</th>
      <th>French Restaurant</th>
      <th>Furniture / Home Store</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>German Restaurant</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Hotel</th>
      <th>Insurance Office</th>
      <th>Italian Restaurant</th>
      <th>Light Rail Station</th>
      <th>Liquor Store</th>
      <th>Mexican Restaurant</th>
      <th>Museum</th>
      <th>Nightclub</th>
      <th>Optical Shop</th>
      <th>Pet Store</th>
      <th>Pier</th>
      <th>Plaza</th>
      <th>Pool</th>
      <th>Pub</th>
      <th>Residential Building (Apartment / Condo)</th>
      <th>Restaurant</th>
      <th>Sandwich Place</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shopping Mall</th>
      <th>Soccer Field</th>
      <th>Spa</th>
      <th>Steakhouse</th>
      <th>Supermarket</th>
      <th>Taverna</th>
      <th>Tea Room</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Trail</th>
      <th>Trattoria/Osteria</th>
      <th>Turkish Restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Croningstraße Wedel</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0625</td>
      <td>0</td>
      <td>0.125</td>
      <td>0</td>
      <td>0</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0625</td>
      <td>0</td>
      <td>0.0625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.1875</td>
      <td>0.0625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Analysis plan<a name="setup"></a>
The general plan follows:
- Retrieve a list of the playgrounds in the vicinity of a German city.
- Use that list to then look up each playground's detailed information.
- Use Foursquare's api to then find which venues are nearby and add to the dataset.
- Find the commercial characteristics of each playground's neighborhood.
- Cluster the playgrounds based on their commercial surroundings.
- Also cluster the playgrounds based on the equipment available.
- Finally, make a few lists of playgrounds with kid-friendly food and icecream nearby and certain playground features.

#### Import the relevant libraries:


```python
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import folium
from folium import plugins
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import matplotlib.colors as colors
from IPython.display import display
pd.set_option('display.max_columns', None)
```

# Part I: Data preparation: <a name="part1"></a>

### Part IA: Get a list of playgrounds in a city of interest <a name="part1A"></a>
Save as a list of URLs linking to detailed playground information

#### Set up the search URL by specifying the city where we want to look for playgrounds:


```python
'''
Note this notebook was designed to query the playgrounds in the vicinities of Elmshorn and Wedel.
For the purposes of this assignment, Wedel is used as the data is particularly complete.
'''
search_city = input("Please enter a German city: ")
print('')
print("If the search does not return a list of playgrounds in the city, please enter the name of a larger city.")
#This is the website url where we can then append a city name and search:
search_site_url = "https://spielplatznet.de/spielplaetze/"
url = search_site_url + search_city
print('')
print("Here is the url where more information is available on the playgrounds available:")
print(url)
```

    Please enter a German city: Wedel
    
    If the search does not return a list of playgrounds in the city, please enter the name of a larger city.
    
    Here is the url where more information is available on the playgrounds available:
    https://spielplatznet.de/spielplaetze/Wedel
    

#### Import the city playground search data using a get request and make it more readable with Beautiful Soup:


```python
data = requests.get(url).text
#Create a soup object using the variable 'data':
soup = BeautifulSoup(data,"html5lib")
#Display the material available to work with:
print(soup.prettify())
```

    <html class="no-js" lang="de">
     <head>
     </head>
     <body class="blau" onload="initialize('Wedel',-1,51.158592,10.406305,10,1,53.5989169842047,9.73169803619385,53.5669774121414,9.68239367008209); ">
      ﻿
      <meta content="ie=edge" http-equiv="x-ua-compatible"/>
      <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
      <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
      <meta content="text/javascript" http-equiv="Content-Script-Type"/>
      <meta content="text/css" http-equiv="Content-Style-Type"/>
      <meta content="german" http-equiv="content-language"/>
      <meta content="public,no-cache" http-equiv="Cache-Control"/>
      <meta content="de" name="Content-Language"/>
      <meta content="german" name="Language"/>
      <meta content="all" name="audience"/>
      <meta content="Spielplatznet.de" name="copyright"/>
      <meta content="INDEX,FOLLOW" name="Robots"/>
      <meta content="0" name="expires"/>
      <meta content="7 days" name="revisit-after"/>
      <meta content="Ralph Anthes" name="author"/>
      <meta content="Spielplatznet.de" name="publisher"/>
      <meta content="Ralph Anthes" name="copyright"/>
      <meta content="DE" name="geo.region"/>
      <meta content="Wedel" name="geo.placename"/>
      <link href="/spielplatznet.xml" rel="alternate" title="Spielplatznet" type="application/rss+xml"/>
      <meta content="#317EFB" name="theme-color"/>
      <link href="/manifest.json" rel="manifest"/>
      <meta content="51 Spielplätze in Wedel auf Spielplatznet.de eingetragen. Spielplätze betrachten, bewerten, beschreiben in Deutschlands großen Spielplatz-Katalog." name="description"/>
      <meta content="Spielplätze,Spielplatz,Wedel,Appelboomtwiete Ecke Steinberg,Albert-Schweizer Schule,Alter Zirkusplatz,Altstadtschule,Anne-Frank-Weg,Ansgariusweg,Autal,Brombeerweg,Bürgerpark,Croningstraße,Gärtnerstraße,Egenbüttelweg,Elbstraße,Ernst-Thälmann-Weg,Geesthang,Gerhart-Hauptmann Straße,Ginsterweg,Hainbuchenweg,Hamburger Yachthafen,Hans-Böckler Platz,Heinrich-Schacht-Straße,Haselweg,Hellgrund,Im Grund,Klintkamp,Kronskamp,Lindenstraße,Meisenweg,Mühlenweg,Opn Klint,Parnaßstraße,Pferdekoppel,Pinneberger Straße,Pulverstraße,Rebhuhnweg,Reepschlägerstraße,Rotdornstraße,Rosengarten,Schlehdornweg,Schwartenseekamp,Strandbad,Vogt-Körner Straße,Von-Suttner Straße,Wacholderstraße,Waldspielplatz Moorwegsiedlung,Wiedetwiete,Tinsdaler Weg,Theaterstraße,Heinestraße,Kids and Play Wedel,Wasserspielplatz Haus am See,Appelboomtwiete Ecke Aastwiete, Spielplatznet" name="keywords"/>
      <meta content="Spielplätze,Spielplatz,Wedel,Appelboomtwiete Ecke Steinberg,Albert-Schweizer Schule,Alter Zirkusplatz,Altstadtschule,Anne-Frank-Weg,Ansgariusweg,Autal,Brombeerweg,Bürgerpark,Croningstraße,Gärtnerstraße,Egenbüttelweg,Elbstraße,Ernst-Thälmann-Weg,Geesthang,Gerhart-Hauptmann Straße,Ginsterweg,Hainbuchenweg,Hamburger Yachthafen,Hans-Böckler Platz,Heinrich-Schacht-Straße,Haselweg,Hellgrund,Im Grund,Klintkamp,Kronskamp,Lindenstraße,Meisenweg,Mühlenweg,Opn Klint,Parnaßstraße,Pferdekoppel,Pinneberger Straße,Pulverstraße,Rebhuhnweg,Reepschlägerstraße,Rotdornstraße,Rosengarten,Schlehdornweg,Schwartenseekamp,Strandbad,Vogt-Körner Straße,Von-Suttner Straße,Wacholderstraße,Waldspielplatz Moorwegsiedlung,Wiedetwiete,Tinsdaler Weg,Theaterstraße,Heinestraße,Kids and Play Wedel,Wasserspielplatz Haus am See,Appelboomtwiete Ecke Aastwiete, Spielplatznet" name="DC.subject"/>
      <meta content="605569D1846D91F4E60BE9D6CBF229C3" name="msvalidate.01"/>
      <title>
       Spielplätze in Wedel | spielplatznet.de
      </title>
      <script async="" src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js">
      </script>
      <script>
       (adsbygoogle=window.adsbygoogle||[]).push({google_ad_client:"ca-pub-3656668926056704",enable_page_level_ads:true});
      </script>
      <script src="/js/leaflet.js.pagespeed.jm.2-9VCX4gWh.js">
      </script>
      <script async="" defer="">
       var pxx=[49.913,49.419,49.192,48.405,48.393,54.381,51.518,50.067,49.021,50.670,51.563,49.343,48.832,48.765,51.084,50.357,52.024,48.438,50.680,51.513,51.468,51.262,49.492,48.457,51.291,48.768,52.372,53.432,50.831,51.060,50.146,51.678,50.080,51.168,49.447,51.206,49.889,49.064,50.926,47.883,51.344,47.840,50.665,50.976,49.812,50.925,50.756,48.451,50.066,52.239,49.912,53.981,51.235,53.623,47.958,47.999,52.286,48.777,53.337,54.162,47.852,49.436,50.810,47.753,51.850,49.464,48.906,51.465,49.422,53.078,50.594,48.483,50.271,51.466,51.216,52.513,52.493,53.318,54.208,50.134,52.207,52.808,50.578,52.564,52.457,52.414,52.309,48.122,53.748,52.173,53.450,48.204,53.563,52.893,51.768,49.500];var pyy=[10.024,10.875,9.370,11.612,10.856,9.903,7.088,9.038,8.348,7.063,7.987,7.106,9.146,9.713,13.763,7.544,7.356,9.010,8.655,7.377,6.522,9.307,8.177,10.083,12.570,13.071,9.736,9.937,6.244,10.897,8.289,6.758,10.978,6.504,11.924,6.861,7.900,12.210,6.842,7.874,7.572,9.648,9.714,7.194,8.714,7.977,11.733,12.374,11.831,10.542,6.691,10.786,7.132,10.038,8.722,11.342,9.793,9.180,7.874,9.243,12.252,11.066,13.003,10.388,8.572,8.528,8.727,9.979,8.732,8.833,12.423,8.004,8.672,11.871,14.572,13.398,7.751,8.828,12.454,8.667,8.708,10.221,10.716,13.351,13.453,13.047,14.133,11.574,13.570,11.769,11.477,11.529,9.994,13.229,14.328,11.741];var pname=[97,91,74,85,86,24,45,63,76,53,59,66,71,73,1,56,48,72,35,44,47,34,67,89,4,94,30,21,52,99,65,46,96,41,92,40,55,93,50,79,58,88,36,51,64,57,7,84,95,38,54,23,42,22,78,82,31,70,26,25,83,90,9,87,33,68,75,37,69,28,8,77,61,6,2,10,49,27,18,60,32,29,98,13,12,14,15,81,17,39,19,80,20,16,3,2147483];var panz=[1885,1762,1656,1654,1492,1477,1433,1387,1384,1378,1307,1258,1256,1212,1191,1187,1169,1166,1141,1133,1124,1102,1098,1075,1060,1052,1022,1017,999,988,986,984,981,978,963,957,950,943,924,917,912,910,894,892,876,857,853,837,814,810,782,781,770,745,740,735,729,700,684,675,665,647,645,644,632,615,605,604,598,572,537,535,477,474,473,465,457,453,412,405,368,364,343,329,312,306,238,236,233,209,199,136,107,106,67,1];
      </script>
      <script async="" defer="" src="/js/jsspn/spielplaetze_small_kat.js.pagespeed.jm.0cnJ-Im41j.js">
      </script>
      <script async="" defer="" src="/js/jsspn/staedte.js.pagespeed.jm.i58nSv_Hps.js">
      </script>
      <script async="" defer="" src="/js/jsspn/osm.json">
      </script>
      <script async="" defer="" src="/js/leafletv2.js.pagespeed.jm.0Ej6RReSmJ.js">
      </script>
      <script async="" defer="" src="/js/gpsload.js.pagespeed.jm.40-6kUq1zP.js">
      </script>
      <script data-pagespeed-no-defer="">
       (function(){function d(b){var a=window;if(a.addEventListener)a.addEventListener("load",b,!1);else if(a.attachEvent)a.attachEvent("onload",b);else{var c=a.onload;a.onload=function(){b.call(this);c&&c.call(this)}}}var p=Date.now||function(){return+new Date};window.pagespeed=window.pagespeed||{};var q=window.pagespeed;function r(){this.a=!0}r.prototype.c=function(b){b=parseInt(b.substring(0,b.indexOf(" ")),10);return!isNaN(b)&&b<=p()};r.prototype.hasExpired=r.prototype.c;r.prototype.b=function(b){return b.substring(b.indexOf(" ",b.indexOf(" ")+1)+1)};r.prototype.getData=r.prototype.b;r.prototype.f=function(b){var a=document.getElementsByTagName("script"),a=a[a.length-1];a.parentNode.replaceChild(b,a)};r.prototype.replaceLastScript=r.prototype.f;
    r.prototype.g=function(b){var a=window.localStorage.getItem("pagespeed_lsc_url:"+b),c=document.createElement(a?"style":"link");a&&!this.c(a)?(c.type="text/css",c.appendChild(document.createTextNode(this.b(a)))):(c.rel="stylesheet",c.href=b,this.a=!0);this.f(c)};r.prototype.inlineCss=r.prototype.g;
    r.prototype.h=function(b,a){var c=window.localStorage.getItem("pagespeed_lsc_url:"+b+" pagespeed_lsc_hash:"+a),f=document.createElement("img");c&&!this.c(c)?f.src=this.b(c):(f.src=b,this.a=!0);for(var c=2,k=arguments.length;c<k;++c){var g=arguments[c].indexOf("=");f.setAttribute(arguments[c].substring(0,g),arguments[c].substring(g+1))}this.f(f)};r.prototype.inlineImg=r.prototype.h;
    function t(b,a,c,f){a=document.getElementsByTagName(a);for(var k=0,g=a.length;k<g;++k){var e=a[k],m=e.getAttribute("data-pagespeed-lsc-hash"),h=e.getAttribute("data-pagespeed-lsc-url");if(m&&h){h="pagespeed_lsc_url:"+h;c&&(h+=" pagespeed_lsc_hash:"+m);var l=e.getAttribute("data-pagespeed-lsc-expiry"),l=l?(new Date(l)).getTime():"",e=f(e);if(!e){var n=window.localStorage.getItem(h);n&&(e=b.b(n))}e&&(window.localStorage.setItem(h,l+" "+m+" "+e),b.a=!0)}}}
    function u(b){t(b,"img",!0,function(a){return a.src});t(b,"style",!1,function(a){return a.firstChild?a.firstChild.nodeValue:null})}
    q.i=function(){if(window.localStorage){var b=new r;q.localStorageCache=b;d(function(){u(b)});d(function(){if(b.a){for(var a=[],c=[],f=0,k=p(),g=0,e=window.localStorage.length;g<e;++g){var m=window.localStorage.key(g);if(!m.indexOf("pagespeed_lsc_url:")){var h=window.localStorage.getItem(m),l=h.indexOf(" "),n=parseInt(h.substring(0,l),10);if(!isNaN(n))if(n<=k){a.push(m);continue}else if(n<f||!f)f=n;c.push(h.substring(l+1,h.indexOf(" ",l+1)))}}k="";f&&(k="; expires="+(new Date(f)).toUTCString());document.cookie=
    "_GPSLSC="+c.join("!")+k;g=0;for(e=a.length;g<e;++g)window.localStorage.removeItem(a[g]);b.a=!1}})}};q.localStorageCacheInit=q.i;})();
    pagespeed.localStorageCacheInit();
      </script>
      <link href="/css/A.foundation.css.pagespeed.cf.SXrIwPuNF-.css" rel="stylesheet"/>
      <link href="/css/app.css" rel="stylesheet"/>
      <link href="/css/A.spielplatznet.css+leaflet.css+cookieconsent.min.css,Mcc.SbPce72TwS.css.pagespeed.cf.oWhRjGwpeq.css" rel="stylesheet"/>
      <img alt="Spielplatznet Logo" class="hide-for-small-only" onclick="location.href='/index.htm';" src="/img/190xNxlogo.png.pagespeed.ic.nYBH8KEVwf.png" style="position:relative;z-index:1;width:190px;top:1px;margin-right:15px;margin-bottom:50 px;cursor: pointer;" title="Zur Startseite"/>
      <img alt="Spielplatznet Logo" class="show-for-small-only" onclick="location.href='/index.htm';" src="/img/130xNxlogo.png.pagespeed.ic.OEd4dz88zp.png" style="position:relative;z-index:1;width:130px;top:1px;margin-right:15px;cursor: pointer;" title="Zur Startseite"/>
      <div class="row">
       <div class="large-12 medium-12 columns">
        <div class="hide-for-small-only">
         <div style="position:absolute;left:210px;top:30px;">
          <div class="button-group radius">
           <a class="button gruen_tief btn_menue" href="/finden.htm" title="Spielplätze finden">
            Spielplätze finden
           </a>
           <a class="button gruen_tief btn_menue" href="/eintragen.htm" title="Spielplatz eintragen">
            Spielplatz eintragen
           </a>
           <a class="button gruen_tief btn_menue" href="/karte.htm" title="Große Karte mit allen Spielplätzen">
            Spielplatzkarte
           </a>
           <a class="button gruen_tief btn_menue" href="/blog/index.php" title="Blog von Spielplatznet">
            Blog
           </a>
           <a class="button gruen_tief btn_menue" href="/feedback.htm" title="Feedback geben">
            Kontakt
           </a>
           <a class="button gruen_tief btn_menue" href="/hilfe.htm" title="Fragen und Antworten">
            FAQ
           </a>
          </div>
         </div>
        </div>
        <div class="show-for-small-only">
         <div style="position:absolute;left:145px;top:10px;font-size:0.9em;">
          <div class="button-group radius">
           <a class="button gruen_tief btn_menue" href="/finden.htm" title="Spielplätze finden">
            Suche
           </a>
           <a class="button gruen_tief btn_menue" href="/eintragen.htm" title="Spielplatz eintragen">
            Neu
           </a>
           <a class="button gruen_tief btn_menue" href="/karte.htm" title="Karte mit allen Spielplätzen">
            Karte
           </a>
          </div>
         </div>
        </div>
       </div>
      </div>
      <div class="row" style="float:left;margin-top:-20px;">
       <div id="loginbutton" style="position:relative;max-height:15px;right:0px;text-align:right;z-index:100;">
        <div class="button btn2 weiss userbtn" data-toggle="reveal_login" id="loginbutton">
         Anmelden
        </div>
        <div class="tiny reveal callout" data-reveal="" id="reveal_login">
         <button aria-label="Close reveal" class="close-button" data-close="" type="button">
          <span aria-hidden="true">
           ×
          </span>
         </button>
         <h3 style="text-align: left;">
          Anmelden
         </h3>
         <form action="/anmeldung.htm" id="anmeldung">
          <input class="text" id="useremail" name="useremail" placeholder="E-Mailadresse" type="text"/>
          <input class="text" id="userpasswort" name="userpasswort" placeholder="Passwort" type="password"/>
          <input name="refer" type="hidden" value="/spielplaetze/Wedel"/>
          <img alt="Go" onclick='document.getElementById("anmeldung").submit();' src="/img/xbutton_pfeil3.png.pagespeed.ic.CkJ0VaQuPy.png" style="cursor: pointer;float: right;"/>
         </form>
         <a href="/registrierung.htm" style="float: left;font-size:0.7em;" title="Hier registrieren...">
          Zur Registrierung...
         </a>
         <br/>
         <hr/>
         <div style="text-align: left;">
          <a href="/sociallogin/index.php?ref=https://spielplatznet.de/spielplaetze/Wedel&amp;fb_login=1" style="text-align: center;">
           <img alt="Mittels Facebook einloggen" src="/images/xfacebook_login.png.pagespeed.ic.y141nYsaYN.png" title="Über Facebook einloggen"/>
          </a>
          <br/>
          <br/>
          <a href="/sociallogin/index.php?ref=https://spielplatznet.de/spielplaetze/Wedel&amp;google_login=1" style="text-align: center;">
           <img alt="Mittels Goolge einloggen" src="/images/xgoogle_login.png.pagespeed.ic.cRwkceXssB.png" title="Über Google einloggen"/>
          </a>
         </div>
        </div>
       </div>
       <div class="weiss">
        <div class="menueleiste">
         <a href="/index.htm" title="Startseite">
          Start
         </a>
         &gt;
         <a href="/finden.htm" title="Spielplatzsuche">
          Suche
         </a>
         &gt;
         <a href="/spielplaetze/" title="Suche nach Spielplätzen">
          Spielplätze
         </a>
         &gt;
         <a href="/spielplaetze/Wedel" title="Spielplätze in Wedel">
          Wedel
         </a>
        </div>
        <h1>
         Spielplatz in Wedel finden
        </h1>
        <div class="full reveal" data-reveal="" id="reveal_filterresults">
         <form action="/finden.htm" id="spielplatz_filter">
          <div class="large-12 medium-12 small-12 columns">
           <h2>
            Filteroptionen
           </h2>
           <input name="Ort" placeholder="Ort oder Postleitzahl" type="text" value="Wedel"/>
           <input name="filtersuche" type="hidden" value="start"/>
           <input class="button btn small show-for-small-only" type="submit" value="Filtersuche starten"/>
          </div>
          <div class="large-3 medium-6 small-12 columns">
           <h3>
            Kategorie
           </h3>
           <input id="spielplatz" name="spielplatz" type="checkbox"/>
           <label for="spielplatz">
            Öffentlicher Spielplatz
           </label>
           <br/>
           <input id="indoor" name="indoor" type="checkbox"/>
           <label for="indoor">
            Indoorspielplatz
           </label>
           <br/>
           <input id="bolzplatz" name="bolzplatz" type="checkbox"/>
           <label for="bolzplatz">
            Ballplatz
           </label>
           <br/>
           <input id="skaten" name="skaten" type="checkbox"/>
           <label for="skaten">
            Skateanlage
           </label>
           <br/>
           <input id="abenteuer" name="abenteuer" type="checkbox"/>
           <label for="abenteuer">
            Abenteuerspielplatz
           </label>
           <br/>
           <input id="kletterwald" name="kletterwald" type="checkbox"/>
           <label for="kletterwald">
            Kletterpark/ -wald
           </label>
           <br/>
           <input id="erlebnisraum" name="erlebnisraum" type="checkbox"/>
           <label for="erlebnisraum">
            Erlebnisraum
           </label>
           <br/>
           <input id="freizeitpark" name="freizeitpark" type="checkbox"/>
           <label for="freizeitpark">
            Freizeitpark
           </label>
           <br/>
           <input id="mehrgenerationen" name="mehrgenerationen" type="checkbox"/>
           <label for="mehrgenerationen">
            Mehrgenerationen
           </label>
           <br/>
           <input id="geburtstag" name="geburtstag" type="checkbox"/>
           <label for="geburtstag">
            Geburtstagsfeier
           </label>
           <br/>
           <input id="rodelberg" name="rodelberg" type="checkbox"/>
           <label for="rodelberg">
            Rodelberg
           </label>
           <br/>
           <input id="wasserspiel" name="wasserspiel" type="checkbox"/>
           <label for="wasserspiel">
            Wasserspiel
           </label>
           <br/>
          </div>
          <div class="large-3 medium-6 small-12 columns">
           <h3>
            Features
           </h3>
           <input id="sterne5" name="sterne5" type="checkbox"/>
           <label for="sterne5">
            5 Sterne
           </label>
           <br/>
           <input id="sterne4" name="sterne4" type="checkbox"/>
           <label for="sterne4">
            4 Sterne
           </label>
           <br/>
           <input id="sterne3" name="sterne3" type="checkbox"/>
           <label for="sterne3">
            3 Sterne
           </label>
           <br/>
           <input id="sterne2" name="sterne2" type="checkbox"/>
           <label for="sterne2">
            2 Sterne
           </label>
           <br/>
           <input id="sterne1" name="sterne1" type="checkbox"/>
           <label for="sterne1">
            1 Stern
           </label>
           <br/>
           <input id="sterne0" name="sterne0" type="checkbox"/>
           <label for="sterne0">
            Ohne Bewertung
           </label>
           <br/>
           <input id="alter0-3" name="alter0-3" type="checkbox"/>
           <label for="alter0-3">
            1-3 Jahre
           </label>
           <br/>
           <input id="alter3-6" name="alter3-6" type="checkbox"/>
           <label for="alter3-6">
            3-6 Jahre
           </label>
           <br/>
           <input id="alter6-12" name="alter6-12" type="checkbox"/>
           <label for="alter6-12">
            6-12 Jahre
           </label>
           <br/>
           <input id="alter12-16" name="alter12-16" type="checkbox"/>
           <label for="alter12-16">
            12-16 Jahre
           </label>
           <br/>
           <input id="behindertengerecht" name="behindertengerecht" type="checkbox"/>
           <label for="behindertengerecht">
            Behindertengerecht
           </label>
           <br/>
           <input id="schatten" name="schatten" type="checkbox"/>
           <label for="schatten">
            Schatten
           </label>
           <br/>
           <input id="gastronomie" name="gastronomie" type="checkbox"/>
           <label for="gastronomie">
            Gastronomie
           </label>
           <br/>
           <input id="wc" name="wc" type="checkbox"/>
           <label for="wc">
            Toilette
           </label>
           <br/>
           <input id="eingefriedet" name="eingefriedet" type="checkbox"/>
           <label for="eingefriedet">
            Eingezäunt
           </label>
           <br/>
           <input id="kostenpflichtig" name="kostenpflichtig" type="checkbox"/>
           <label for="kostenpflichtig">
            Kostenpflichtig
           </label>
           <br/>
           <input class="button btn small show-for-small-only" type="submit" value="Filtersuche starten"/>
          </div>
          <div class="large-3 medium-6 small-12 columns">
           <h3>
            Spielgeräte Action
           </h3>
           <input id="seilbahn" name="seilbahn" type="checkbox"/>
           <label for="seilbahn">
            Seilbahn
           </label>
           <br/>
           <input id="karussell" name="karussell" type="checkbox"/>
           <label for="karussell">
            Karussell
           </label>
           <br/>
           <input id="trampolin" name="trampolin" type="checkbox"/>
           <label for="trampolin">
            Trampolin
           </label>
           <br/>
           <input id="schaukel" name="schaukel" type="checkbox"/>
           <label for="schaukel">
            Schaukel
           </label>
           <br/>
           <input id="kletterwand" name="kletterwand" type="checkbox"/>
           <label for="kletterwand">
            Kletterwand
           </label>
           <br/>
           <input id="tore" name="tore" type="checkbox"/>
           <label for="tore">
            Tore
           </label>
           <br/>
           <input id="basketball" name="basketball" type="checkbox"/>
           <label for="basketball">
            Basketballkorb
           </label>
           <br/>
           <input id="tischtennis" name="tischtennis" type="checkbox"/>
           <label for="tischtennis">
            Tischtennisplatte
           </label>
           <br/>
           <input id="reckstange" name="reckstange" type="checkbox"/>
           <label for="reckstange">
            Reckstange
           </label>
           <br/>
          </div>
          <div class="large-3 medium-6 small-12 columns">
           <h3>
            Spielgeräte Kleinkind
           </h3>
           <input id="rutsche" name="rutsche" type="checkbox"/>
           <label for="rutsche">
            Rutsche
           </label>
           <br/>
           <input id="sandkasten" name="sandkasten" type="checkbox"/>
           <label for="sandkasten">
            Sandspielbereich
           </label>
           <br/>
           <input id="spielhaus" name="spielhaus" type="checkbox"/>
           <label for="spielhaus">
            Spielhaus
           </label>
           <br/>
           <input id="nestschaukel" name="nestschaukel" type="checkbox"/>
           <label for="nestschaukel">
            Nestschaukel
           </label>
           <br/>
           <input id="babyschaukel" name="babyschaukel" type="checkbox"/>
           <label for="babyschaukel">
            Babyschaukel
           </label>
           <br/>
           <input id="klettergeraet" name="klettergeraet" type="checkbox"/>
           <label for="klettergeraet">
            Klettergerät
           </label>
           <br/>
           <input id="wippe" name="wippe" type="checkbox"/>
           <label for="wippe">
            Wippe/Federtier
           </label>
           <br/>
          </div>
          <div class="large-12 medium-12 small-12 columns">
           <input class="button btn small" type="submit" value="Filtersuche starten"/>
          </div>
         </form>
         <button aria-label="Filtersuche schließen" class="close-button" data-close="" type="button">
          <span aria-hidden="true">
           ×
          </span>
         </button>
        </div>
        <div class="large-6 medium-6 columns">
         <div class="callout gruen_tief">
          <form action="/finden.htm" id="spielplatz" method="post">
           <label class="gruen_tief">
            Wählen Sie den Ort oder die Postleitzahl
           </label>
           <div style="white-space: nowrap;width:95%;">
            <input name="Ort" placeholder="Ort oder Postleitzahl" style="width:50%" type="text" value="Wedel"/>
            <select id="select" name="cat" required="" style="width:40%">
             <option value="Spielplatz">
              Spielplätze
             </option>
             <option value="Alle Typen">
              Alle Kategorien
             </option>
             <option value="Indoorspielplatz">
              Indoorspielplätze
             </option>
             <option value="Wasserspielplatz">
              Wasserspielplätze
             </option>
             <option value="Abenteuerspielplatz">
              Abenteuerspielplätze
             </option>
             <option value="Ballplatz">
              Ballplätze
             </option>
             <option value="Rodelberg">
              Rodelberge
             </option>
             <option value="Kletterparks">
              Kletterparks
             </option>
             <option value="Skateplatz">
              Skateplätze
             </option>
             <option value="Freizeitpark">
              Freizeitparks
             </option>
             <option value="Tischtennisplatte">
              Tischtennisplatte
             </option>
             <option value="Kindergeburtstag">
              Kindergeburtstage
             </option>
             <option value="Mehrgenerationenspielplatz">
              Mehrgenerationenspielplätze
             </option>
            </select>
            <button id="berechnen" type="submit">
             <img alt="Go" height="36" name="bild1" src="/img/xbutton_pfeil.png.pagespeed.ic.raMqIQ5sxZ.png" style="cursor: pointer;" width="37"/>
            </button>
           </div>
           <label class="gruen_tief" data-toggle="reveal_filterresults" style="cursor: pointer;">
            Suche filtern...
           </label>
          </form>
         </div>
         <div class="callout" style="border:none;margin:0px;padding:0px;overflow:hidden;">
          <ins class="adsbygoogle" data-ad-client="ca-pub-3656668926056704" data-ad-format="auto" data-ad-slot="3992555449" data-full-width-responsive="true" style="display:block">
          </ins>
          <script>
           (adsbygoogle=window.adsbygoogle||[]).push({});
          </script>
         </div>
         <div style="margin-top: 5px; right: 0px; position: relative; float: right; width: 160px;">
          <iframe allowtransparency="true" frameborder="0" scrolling="no" src="//www.facebook.com/plugins/like.php?href=https%3A%2F%2Fwww.spielplatznet.de%2Fspielplaetze%2FWedel&amp;width&amp;layout=button_count&amp;action=like&amp;show_faces=false&amp;share=true&amp;height=21&amp;appId=178180998924296" style="border:none; overflow:hidden; height:21px;">
          </iframe>
         </div>
         <h3>
          51 Spielplätze in Wedel gefunden
         </h3>
         <div style="width:100%;">
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(872)" onmouseover="showStandort(872,53.5926308917772,9.73169803619385)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/872/Wedel/Waldspielplatz Moorwegsiedlung">
            Waldspielplatz Moorwegsiedlung
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5926308917772,9.73169803619385,16)">
           <img alt="5 Sterne bei 4 Bewertungen" height="12" src="https://spielplatznet.de/images/bewertung/70x12x50.png.pagespeed.ic.aO6x-yYj9L.jpg" title="5 Sterne bei 4 Bewertungen" width="70"/>
          </div>
          <div class="zelle_right">
           <img alt="Fotos vorhanden" src="/images/fotocam.jpg.pagespeed.ce.TsdKkbwOnj.jpg" title="Spielplatz mit Foto"/>
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(849)" onmouseover="showStandort(849,53.5912910844463,9.70646917819977)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/849/Wedel/Haselweg">
            Haselweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5912910844463,9.70646917819977,16)">
           <img alt="5 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x50.png.pagespeed.ic.aO6x-yYj9L.jpg" title="5 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(855)" onmouseover="showStandort(855,53.5943062279335,9.71506834030151)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/855/Wedel/Meisenweg">
            Meisenweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5943062279335,9.71506834030151,16)">
           <img alt="5 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x50.png.pagespeed.ic.aO6x-yYj9L.jpg" title="5 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(17868)" onmouseover="showStandort(17868,53.5914407324793,9.70553040504456)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/17868/Wedel/Wasserspielplatz Haus am See">
            Wasserspielplatz Haus am See
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5914407324793,9.70553040504456,16)">
           <img alt="5 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x50.png.pagespeed.ic.aO6x-yYj9L.jpg" title="5 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(856)" onmouseover="showStandort(856,53.581066013162,9.71019208431244)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/856/Wedel/Mühlenweg">
            Mühlenweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.581066013162,9.71019208431244,16)">
           <img alt="4.5 Sterne bei 2 Bewertungen" height="12" src="https://spielplatznet.de/images/bewertung/70x12x45.png.pagespeed.ic.jiKerCBIw0.jpg" title="4.5 Sterne bei 2 Bewertungen" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(864)" onmouseover="showStandort(864,53.591066930019,9.68836158514023)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/864/Wedel/Rotdornstraße">
            Rotdornstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.591066930019,9.68836158514023,16)">
           <img alt="4.5 Sterne bei 2 Bewertungen" height="12" src="https://spielplatznet.de/images/bewertung/70x12x45.png.pagespeed.ic.jiKerCBIw0.jpg" title="4.5 Sterne bei 2 Bewertungen" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(846)" onmouseover="showStandort(846,53.5743968895724,9.68239367008209)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/846/Wedel/Hamburger Yachthafen">
            Hamburger Yachthafen
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5743968895724,9.68239367008209,16)">
           <img alt="4 Sterne bei 3 Bewertungen" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei 3 Bewertungen" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(844)" onmouseover="showStandort(844,53.5736384685368,9.71911311149597)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/844/Wedel/Ginsterweg">
            Ginsterweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5736384685368,9.71911311149597,16)">
           <img alt="4 Sterne bei 2 Bewertungen" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei 2 Bewertungen" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(847)" onmouseover="showStandort(847,53.5688601987249,9.71491277217865)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/847/Wedel/Hans-Böckler Platz">
            Hans-Böckler Platz
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5688601987249,9.71491277217865,16)">
           <img alt="4 Sterne bei 2 Bewertungen" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei 2 Bewertungen" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(861)" onmouseover="showStandort(861,53.5712051224608,9.71940010786057)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/861/Wedel/Pulverstraße">
            Pulverstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5712051224608,9.71940010786057,16)">
           <img alt="4 Sterne bei 2 Bewertungen" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei 2 Bewertungen" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(830)" onmouseover="showStandort(830,53.5755961290153,9.71072316169739)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/830/Wedel/Alter Zirkusplatz">
            Alter Zirkusplatz
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5755961290153,9.71072316169739,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(831)" onmouseover="showStandort(831,53.582625249582,9.69926744699478)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/831/Wedel/Altstadtschule">
            Altstadtschule
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.582625249582,9.69926744699478,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(832)" onmouseover="showStandort(832,53.5889495035841,9.69376623630524)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/832/Wedel/Anne-Frank-Weg">
            Anne-Frank-Weg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5889495035841,9.69376623630524,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(833)" onmouseover="showStandort(833,53.5871962578912,9.68622386455536)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/833/Wedel/Ansgariusweg">
            Ansgariusweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5871962578912,9.68622386455536,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(835)" onmouseover="showStandort(835,53.5741194511191,9.72591519355774)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/835/Wedel/Brombeerweg">
            Brombeerweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5741194511191,9.72591519355774,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
           <img alt="Fotos vorhanden" src="/images/fotocam.jpg.pagespeed.ce.TsdKkbwOnj.jpg" title="Spielplatz mit Foto"/>
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(837)" onmouseover="showStandort(837,53.5819099697564,9.72337782382965)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/837/Wedel/Croningstraße">
            Croningstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5819099697564,9.72337782382965,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(838)" onmouseover="showStandort(838,53.5856072564417,9.69738721847534)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/838/Wedel/Gärtnerstraße">
            Gärtnerstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5856072564417,9.69738721847534,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
           <img alt="Fotos vorhanden" src="/images/fotocam.jpg.pagespeed.ce.TsdKkbwOnj.jpg" title="Spielplatz mit Foto"/>
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(841)" onmouseover="showStandort(841,53.5881674618245,9.69368040561676)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/841/Wedel/Ernst-Thälmann-Weg">
            Ernst-Thälmann-Weg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5881674618245,9.69368040561676,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(842)" onmouseover="showStandort(842,53.5897862207119,9.68335154549268)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/842/Wedel/Geesthang">
            Geesthang
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5897862207119,9.68335154549268,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(848)" onmouseover="showStandort(848,53.5800213944949,9.72487986087799)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/848/Wedel/Heinrich-Schacht-Straße">
            Heinrich-Schacht-Straße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5800213944949,9.72487986087799,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(854)" onmouseover="showStandort(854,53.5784133245173,9.7196630962585)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/854/Wedel/Lindenstraße">
            Lindenstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5784133245173,9.7196630962585,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
           <img alt="Fotos vorhanden" src="/images/fotocam.jpg.pagespeed.ce.TsdKkbwOnj.jpg" title="Spielplatz mit Foto"/>
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(859)" onmouseover="showStandort(859,53.5889794348674,9.7067803144455)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/859/Wedel/Pferdekoppel">
            Pferdekoppel
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5889794348674,9.7067803144455,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(860)" onmouseover="showStandort(860,53.5855916527196,9.70323175191879)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/860/Wedel/Pinneberger Straße">
            Pinneberger Straße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5855916527196,9.70323175191879,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(865)" onmouseover="showStandort(865,53.5810532740654,9.70545530319214)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/865/Wedel/Rosengarten">
            Rosengarten
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5810532740654,9.70545530319214,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(867)" onmouseover="showStandort(867,53.5989169842047,9.73109203352019)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/867/Wedel/Schwartenseekamp">
            Schwartenseekamp
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5989169842047,9.73109203352019,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(868)" onmouseover="showStandort(868,53.5709480505284,9.69663619995117)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/868/Wedel/Strandbad">
            Strandbad
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5709480505284,9.69663619995117,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(869)" onmouseover="showStandort(869,53.5751228077681,9.70517635345459)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/869/Wedel/Vogt-Körner Straße">
            Vogt-Körner Straße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5751228077681,9.70517635345459,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(871)" onmouseover="showStandort(871,53.5907752727735,9.69387352466583)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/871/Wedel/Wacholderstraße">
            Wacholderstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5907752727735,9.69387352466583,16)">
           <img alt="4 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x40.png.pagespeed.ic.fVhHcms6fr.jpg" title="4 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(853)" onmouseover="showStandort(853,53.5807953065342,9.72207427024841)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/853/Wedel/Kronskamp">
            Kronskamp
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5807953065342,9.72207427024841,16)">
           <img alt="3 Sterne bei 2 Bewertungen" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei 2 Bewertungen" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(828)" onmouseover="showStandort(828,53.5888347076387,9.69773345623709)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/828/Wedel/Appelboomtwiete Ecke Steinberg">
            Appelboomtwiete Ecke Steinberg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5888347076387,9.69773345623709,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(829)" onmouseover="showStandort(829,53.5717208544489,9.72274482250214)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/829/Wedel/Albert-Schweizer Schule">
            Albert-Schweizer Schule
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5717208544489,9.72274482250214,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(836)" onmouseover="showStandort(836,53.5848168731614,9.69250559806824)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/836/Wedel/Bürgerpark">
            Bürgerpark
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5848168731614,9.69250559806824,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(839)" onmouseover="showStandort(839,53.5911799625823,9.72104430198669)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/839/Wedel/Egenbüttelweg">
            Egenbüttelweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5911799625823,9.72104430198669,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
           <img alt="Fotos vorhanden" src="/images/fotocam.jpg.pagespeed.ce.TsdKkbwOnj.jpg" title="Spielplatz mit Foto"/>
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(840)" onmouseover="showStandort(840,53.5699401349262,9.7113025188446)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/840/Wedel/Elbstraße">
            Elbstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5699401349262,9.7113025188446,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(843)" onmouseover="showStandort(843,53.591999437962,9.72143810255561)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/843/Wedel/Gerhart-Hauptmann Straße">
            Gerhart-Hauptmann Straße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.591999437962,9.72143810255561,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
           <img alt="Fotos vorhanden" src="/images/fotocam.jpg.pagespeed.ce.TsdKkbwOnj.jpg" title="Spielplatz mit Foto"/>
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(845)" onmouseover="showStandort(845,53.5899384230329,9.69445219130904)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/845/Wedel/Hainbuchenweg">
            Hainbuchenweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5899384230329,9.69445219130904,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(850)" onmouseover="showStandort(850,53.5669774121414,9.72199380397797)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/850/Wedel/Hellgrund">
            Hellgrund
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5669774121414,9.72199380397797,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(852)" onmouseover="showStandort(852,53.5919426661751,9.71341580374904)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/852/Wedel/Klintkamp">
            Klintkamp
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5919426661751,9.71341580374904,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(857)" onmouseover="showStandort(857,53.5873395516784,9.70869541168213)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/857/Wedel/Opn Klint">
            Opn Klint
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5873395516784,9.70869541168213,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(863)" onmouseover="showStandort(863,53.5863301162117,9.69326734542847)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/863/Wedel/Reepschlägerstraße">
            Reepschlägerstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5863301162117,9.69326734542847,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(866)" onmouseover="showStandort(866,53.5898009446568,9.69020962715149)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/866/Wedel/Schlehdornweg">
            Schlehdornweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5898009446568,9.69020962715149,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(870)" onmouseover="showStandort(870,53.59188443917404,9.724164381623268)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/870/Wedel/Von-Suttner Straße">
            Von-Suttner Straße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.59188443917404,9.724164381623268,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
           <img alt="Fotos vorhanden" src="/images/fotocam.jpg.pagespeed.ce.TsdKkbwOnj.jpg" title="Spielplatz mit Foto"/>
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(873)" onmouseover="showStandort(873,53.5884908446434,9.70395349985231)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/873/Wedel/Wiedetwiete">
            Wiedetwiete
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5884908446434,9.70395349985231,16)">
           <img alt="3 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x30.png.pagespeed.ic.Z-fDxkQh43.jpg" title="3 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(834)" onmouseover="showStandort(834,53.5824138041649,9.71122829167579)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/834/Wedel/Autal">
            Autal
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5824138041649,9.71122829167579,16)">
           <img alt="2 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x20.png.pagespeed.ic.pKx5aU-B99.jpg" title="2 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(851)" onmouseover="showStandort(851,53.575454069497,9.7262316942215)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/851/Wedel/Im Grund">
            Im Grund
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.575454069497,9.7262316942215,16)">
           <img alt="2 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x20.png.pagespeed.ic.pKx5aU-B99.jpg" title="2 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(858)" onmouseover="showStandort(858,53.5695355602878,9.70255315303802)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/858/Wedel/Parnaßstraße">
            Parnaßstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5695355602878,9.70255315303802,16)">
           <img alt="2 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x20.png.pagespeed.ic.pKx5aU-B99.jpg" title="2 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(862)" onmouseover="showStandort(862,53.5933545865536,9.71821457147598)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/862/Wedel/Rebhuhnweg">
            Rebhuhnweg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5933545865536,9.71821457147598,16)">
           <img alt="2 Sterne bei einer Bewertung" height="12" src="https://spielplatznet.de/images/bewertung/70x12x20.png.pagespeed.ic.pKx5aU-B99.jpg" title="2 Sterne bei einer Bewertung" width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(57434)" onmouseover="showStandort(57434,53.59039491694423,9.69691358503951)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/57434/Wedel/Appelboomtwiete Ecke Aastwiete">
            Appelboomtwiete Ecke Aastwiete
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.59039491694423,9.69691358503951,16)">
           <img alt="Leider noch keine Bewertung." height="12" src="https://spielplatznet.de/images/bewertung/70x12x0.png.pagespeed.ic.MLW8xkh-cW.jpg" title="Leider noch keine Bewertung." width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(7088)" onmouseover="showStandort(7088,53.5754046192883,9.71914261579514)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/7088/Wedel/Tinsdaler Weg">
            Tinsdaler Weg
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5754046192883,9.71914261579514,16)">
           <img alt="Leider noch keine Bewertung." height="12" src="https://spielplatznet.de/images/bewertung/70x12x0.png.pagespeed.ic.MLW8xkh-cW.jpg" title="Leider noch keine Bewertung." width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(7130)" onmouseover="showStandort(7130,53.5822166562335,9.70818042755127)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/7130/Wedel/Theaterstraße">
            Theaterstraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5822166562335,9.70818042755127,16)">
           <img alt="Leider noch keine Bewertung." height="12" src="https://spielplatznet.de/images/bewertung/70x12x0.png.pagespeed.ic.MLW8xkh-cW.jpg" title="Leider noch keine Bewertung." width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <div onmouseout="removeStandort(7131)" onmouseover="showStandort(7131,53.5937489838527,9.73056882619858)" style="width:100%;">
          <div class="zelle_left">
           <a href="/spielplatz/7131/Wedel/Heinestraße">
            Heinestraße
           </a>
           ,
           <a href="/spielplaetze/Wedel">
            Wedel
           </a>
          </div>
          <div class="zelle_right" onclick="view(53.5937489838527,9.73056882619858,16)">
           <img alt="Leider noch keine Bewertung." height="12" src="https://spielplatznet.de/images/bewertung/70x12x0.png.pagespeed.ic.MLW8xkh-cW.jpg" title="Leider noch keine Bewertung." width="70"/>
          </div>
          <div class="zelle_right">
          </div>
         </div>
         <div style="clear:both;">
         </div>
         <br/>
         <a class="button btn2" href="/eintragen.htm?Adresse=Wedel " target="_parent">
          Einen weiteren Spielplatz in
          <b>
           Wedel
          </b>
          eintragen...
         </a>
        </div>
        <div class="large-6 medium-6 columns">
         <div class="callout" style="border:none;margin:0px;padding:0px;overflow:hidden;">
         </div>
         <div id="closelocations" style="position:absolute;margin-top:70px;float:auto;width:150px;font-size:0.7em;z-index:100;background-color:#FFF;visibility:hidden;opacity:0.85;">
          Warte auf Daten...
          <!-- br-->
         </div>
         <div id="mapid" style="width: 100%; height: 480px;z-index:60;">
         </div>
         <div style="position:relative;margin-top:-290px;left:100%;width:85px;margin-left:-95px;font-size:0.7em;z-index:100;width:1px;">
          <div class="button rot_tief btn_menue" id="tracing" onclick="toogleTracing()" style="width:85px;z-index:100;">
           Standort zen-
           <br/>
           trieren aus
          </div>
          <div class="button rot_tief btn_menue" id="spnmap" onclick="toogleSpnMap()" style="width:85px;z-index:100;">
           Plätze in der
           <br/>
           Nähe aus
          </div>
          <br/>
          <div class="button rot_tief btn_menue" id="osm" onclick="toogleOSMSpielplaetze()" style="width:85px;z-index:100;">
           OSM Spiel-
           <br/>
           plätze aus
          </div>
          <br/>
          <div class="button gruen_tief btn_menue" id="katbew" onclick="toogleKatBewert(1)" style="width:85px;z-index:100;">
           Kategorien
          </div>
          <div style="height:188px;z-index:-1;width:1px;">
          </div>
         </div>
         <div class="gpsload small" id="GPSload">
          Warte auf GPS-Signal...
         </div>
         <div class="large-12 medium-12 columns">
          <br/>
          <a class="button btn2" href="/karte.htm" title="Wedel auf großer Karte anzeigen...">
           Wedel auf großer Karte anzeigen...
          </a>
         </div>
         <div class="large-6 medium-6 columns">
          <div class="nowrap" style="overflow:hidden;">
           <h3>
            Kategorien Wedel
           </h3>
           <a href="/spielplaetze/Wedel" title="Spielplätze in Wedel und Umgebung">
            51 Spielplätze in Wedel
           </a>
           <br/>
           <a href="/k/bolzplatz/Wedel" title="Ballplätze in Wedel und Umgebung">
            1 Ballplätze in Wedel
           </a>
           <br/>
           <a href="/k/wasserspielplatz/Wedel" title="Wasserspielplätze in Wedel und Umgebung">
            4 Wasserspielplätze in Wedel
           </a>
           <br/>
           <a href="/k/indoorspielplatz/Wedel" title="Indoorspielplätze in Wedel und Umgebung">
            0 Indoorspielplätze in Wedel
           </a>
           <br/>
           <a href="/k/freizeitpark/Wedel" title="Freizeitparks in Wedel und Umgebung">
            0 Freizeitparks in Wedel
           </a>
           <br/>
           <a href="/k/abenteuerspielplatz/Wedel" title="Abenteuerspielplätze in Wedel und Umgebung">
            0 Abenteuerspielplätze in Wedel
           </a>
           <br/>
           <a href="/k/mehrgenerationenspielplatz/Wedel" title="Mehrgenerationensp. in Wedel und Umgebung">
            0 Mehrgenerationensp. in Wedel
           </a>
           <br/>
           <a href="/k/rodelberg/Wedel" title="Rodelberge in Wedel und Umgebung">
            0 Rodelberge in Wedel
           </a>
           <br/>
           <a href="/k/skateplatz/Wedel" title="Skateplätze in Wedel und Umgebung">
            0 Skateplätze in Wedel
           </a>
           <br/>
           <a href="/k/kindergeburtstag/Wedel" title="Kindergeburtstage in Wedel und Umgebung">
            0 Kindergeburtstage in Wedel
           </a>
           <br/>
           <a href="/k/tischtennis/Wedel" title="Tischtennisplatten in Wedel und Umgebung">
            16 Tischtennisplatten in Wedel
           </a>
           <br/>
           <a href="/k/kletterpark/Wedel" title="Kletterparks in Wedel und Umgebung">
            0 Kletterparks in Wedel
           </a>
           <br/>
          </div>
         </div>
         <div class="large-6 medium-6 columns nowrap">
          <h3>
           Orte (Spielplätze) bei Wedel
          </h3>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Holm" title="1 Spielplätze in Holm">
             Holm
            </a>
           </div>
           <div class="zelle35 small">
            4.5 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Grünendeich" title="1 Spielplätze in Grünendeich">
             Grünendeich
            </a>
           </div>
           <div class="zelle35 small">
            5.3 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Jork" title="24 Spielplätze in Jork">
             Jork
            </a>
           </div>
           <div class="zelle35 small">
            6.2 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Mittelnkirchen" title="1 Spielplätze in Mittelnkirchen">
             Mittelnkirchen
            </a>
           </div>
           <div class="zelle35 small">
            7.1 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Steinkirchen" title="1 Spielplätze in Steinkirchen">
             Steinkirchen
            </a>
           </div>
           <div class="zelle35 small">
            7.4 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Heist" title="1 Spielplätze in Heist">
             Heist
            </a>
           </div>
           <div class="zelle35 small">
            8.4 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Pinneberg" title="28 Spielplätze in Pinneberg">
             Pinneberg
            </a>
           </div>
           <div class="zelle35 small">
            9.2 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Hollern-Twielenfleth" title="1 Spielplätze in Hollern-Twielenfleth">
             Hollern-Twielenfleth
            </a>
           </div>
           <div class="zelle35 small">
            9.8 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Halstenbek" title="18 Spielplätze in Halstenbek">
             Halstenbek
            </a>
           </div>
           <div class="zelle35 small">
            10.3 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Rellingen" title="11 Spielplätze in Rellingen">
             Rellingen
            </a>
           </div>
           <div class="zelle35 small">
            11.6 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Dollern" title="3 Spielplätze in Dollern">
             Dollern
            </a>
           </div>
           <div class="zelle35 small">
            11.7 km
           </div>
          </div>
          <div style="width:160px">
           <div class="zelle120">
            <a href="/spielplaetze/Uetersen" title="16 Spielplätze in Uetersen">
             Uetersen
            </a>
           </div>
           <div class="zelle35 small">
            11.9 km
           </div>
          </div>
          <div style="clear:both">
          </div>
         </div>
         <div class="callout" style="border:none;margin:0px;padding:0px;">
         </div>
        </div>
        <script>
         (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){(i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)})(window,document,'script','//www.google-analytics.com/analytics.js','ga');ga('create','UA-4855941-5','spielplatznet.de');ga('require','displayfeatures');ga('send','pageview');setTimeout("ga('send','event','Interessierte Nutzer','Mehr als 10 Sekunden')",10000);
        </script>
       </div>
      </div>
      <div class="row" style="float:left;clear:left;">
       <div class="footer gruen_tief">
        <a href="/feedback.htm">
         Feedback
        </a>
        |
        <a href="/impressum.htm">
         Impressum
        </a>
        |
        <a href="/nutzung.htm">
         Nutzungsbedingungen
        </a>
        |
        <a href="/datenschutz.htm">
         Datenschutz
        </a>
        <br/>
        <div class="small">
         © 2007-2021 Ralph Anthes | Made with ❤ in Hannover
        </div>
       </div>
      </div>
      <script src="/js/jquery.min.js+what-input.min.js.pagespeed.jc.nVxVX7tmmQ.js">
      </script>
      <script>
       eval(mod_pagespeed_ETdxprX6Aj);
      </script>
      <script>
       eval(mod_pagespeed_oxpLTB4zKG);
      </script>
      <script src="/js/foundation.min.js">
      </script>
      <script>
       $(document).foundation()
      </script>
      <script>
       $(document).foundation();$('#reveal_popup').foundation('open');
      </script>
      <div class="hide-for-small-only" style="overflow:hidden;">
      </div>
      <script src="/js/cookieconsent.min.js.pagespeed.ce.QyCaU7OyvN.js">
      </script>
      <script>
       window.addEventListener("load",function(){window.cookieconsent.initialise({"palette":{"popup":{"background":"#383b75"},"button":{"background":"#f1d600"}},"theme":"classic","position":"bottom-right","content":{"message":"Wir verwenden Cookies, um Inhalte zu personalisieren, Funktionen für soziale Medien anbieten zu können und die Zugriffe auf unsere Website zu analysieren. ","dismiss":"Einverstanden","link":"Zur Datenschutzerklärung...","href":"/datenschutz.htm"}})});
      </script>
     </body>
    </html>
    

#### Given a search city, get a list of playground information urls:


```python
#Prepare an empty list:
str_links=[]
#Search through soup for html anchor/links represented by the tag <a>:
for link in soup.findAll('a'):
    #When found, append to the empty list:
    temp_href=link.get('href')
    temp_href.replace('/spielplatz/', '/', 1)
    str_links.append("https://spielplatznet.de"+temp_href)
#I only want the list entries that have url data, so find and keep those:
playground_urls = [s for s in str_links if "/"+search_city+"/" in s]
#Because of the way the source site is written, also need to find and drop entries containing "spielplaetze":
playground_urls = [s for s in playground_urls if "spielplaetze" not in s]
#Check whether the list of playground urls matches the expectation (for Wedel should be 51):
print(len(playground_urls))
playground_urls
```

    51
    




    ['https://spielplatznet.de/spielplatz/872/Wedel/Waldspielplatz Moorwegsiedlung',
     'https://spielplatznet.de/spielplatz/849/Wedel/Haselweg',
     'https://spielplatznet.de/spielplatz/855/Wedel/Meisenweg',
     'https://spielplatznet.de/spielplatz/17868/Wedel/Wasserspielplatz Haus am See',
     'https://spielplatznet.de/spielplatz/856/Wedel/Mühlenweg',
     'https://spielplatznet.de/spielplatz/864/Wedel/Rotdornstraße',
     'https://spielplatznet.de/spielplatz/846/Wedel/Hamburger Yachthafen',
     'https://spielplatznet.de/spielplatz/844/Wedel/Ginsterweg',
     'https://spielplatznet.de/spielplatz/847/Wedel/Hans-Böckler Platz',
     'https://spielplatznet.de/spielplatz/861/Wedel/Pulverstraße',
     'https://spielplatznet.de/spielplatz/830/Wedel/Alter Zirkusplatz',
     'https://spielplatznet.de/spielplatz/831/Wedel/Altstadtschule',
     'https://spielplatznet.de/spielplatz/832/Wedel/Anne-Frank-Weg',
     'https://spielplatznet.de/spielplatz/833/Wedel/Ansgariusweg',
     'https://spielplatznet.de/spielplatz/835/Wedel/Brombeerweg',
     'https://spielplatznet.de/spielplatz/837/Wedel/Croningstraße',
     'https://spielplatznet.de/spielplatz/838/Wedel/Gärtnerstraße',
     'https://spielplatznet.de/spielplatz/841/Wedel/Ernst-Thälmann-Weg',
     'https://spielplatznet.de/spielplatz/842/Wedel/Geesthang',
     'https://spielplatznet.de/spielplatz/848/Wedel/Heinrich-Schacht-Straße',
     'https://spielplatznet.de/spielplatz/854/Wedel/Lindenstraße',
     'https://spielplatznet.de/spielplatz/859/Wedel/Pferdekoppel',
     'https://spielplatznet.de/spielplatz/860/Wedel/Pinneberger Straße',
     'https://spielplatznet.de/spielplatz/865/Wedel/Rosengarten',
     'https://spielplatznet.de/spielplatz/867/Wedel/Schwartenseekamp',
     'https://spielplatznet.de/spielplatz/868/Wedel/Strandbad',
     'https://spielplatznet.de/spielplatz/869/Wedel/Vogt-Körner Straße',
     'https://spielplatznet.de/spielplatz/871/Wedel/Wacholderstraße',
     'https://spielplatznet.de/spielplatz/853/Wedel/Kronskamp',
     'https://spielplatznet.de/spielplatz/828/Wedel/Appelboomtwiete Ecke Steinberg',
     'https://spielplatznet.de/spielplatz/829/Wedel/Albert-Schweizer Schule',
     'https://spielplatznet.de/spielplatz/836/Wedel/Bürgerpark',
     'https://spielplatznet.de/spielplatz/839/Wedel/Egenbüttelweg',
     'https://spielplatznet.de/spielplatz/840/Wedel/Elbstraße',
     'https://spielplatznet.de/spielplatz/843/Wedel/Gerhart-Hauptmann Straße',
     'https://spielplatznet.de/spielplatz/845/Wedel/Hainbuchenweg',
     'https://spielplatznet.de/spielplatz/850/Wedel/Hellgrund',
     'https://spielplatznet.de/spielplatz/852/Wedel/Klintkamp',
     'https://spielplatznet.de/spielplatz/857/Wedel/Opn Klint',
     'https://spielplatznet.de/spielplatz/863/Wedel/Reepschlägerstraße',
     'https://spielplatznet.de/spielplatz/866/Wedel/Schlehdornweg',
     'https://spielplatznet.de/spielplatz/870/Wedel/Von-Suttner Straße',
     'https://spielplatznet.de/spielplatz/873/Wedel/Wiedetwiete',
     'https://spielplatznet.de/spielplatz/834/Wedel/Autal',
     'https://spielplatznet.de/spielplatz/851/Wedel/Im Grund',
     'https://spielplatznet.de/spielplatz/858/Wedel/Parnaßstraße',
     'https://spielplatznet.de/spielplatz/862/Wedel/Rebhuhnweg',
     'https://spielplatznet.de/spielplatz/57434/Wedel/Appelboomtwiete Ecke Aastwiete',
     'https://spielplatznet.de/spielplatz/7088/Wedel/Tinsdaler Weg',
     'https://spielplatznet.de/spielplatz/7130/Wedel/Theaterstraße',
     'https://spielplatznet.de/spielplatz/7131/Wedel/Heinestraße']



### Part IB: Get detailed playground information for each location <a name="part1B"></a>
Transform the playground information into a dataframe.
<br /> Note the source is an amateur, crowd-sourced site. 
So, getting the relevant information out of it is less straightforward than some of those used in the course.
<br />
<br />In this section I primarily rely on writing functions to do the work, then calling them at the end.

#### A function that retrieves a playground's name and geocoordinates:


```python
'''
This function pulls the basic descriptive data for a playground. It takes a playground's url data as 'a_soup'
and returns the playground's name, latitude, longitude, and a longer description that sometimes includes the 
street address. The longer description is crowd-sourced and so is a bit inconsistent. The output is a dictionary
with entries {a_name: playground name, a_lat: latitude, a_long: longitude, a_name_address: long description}
'''
def name_location_data(a_soup):
    #Parse out the playground's name:
    a_name=str(a_soup.find_all('meta',property="og:site_name"))
    a_name=a_name.split('"')
    a_name=a_name[1].split(':')
    a_name=[a_name[1]]
    #Parse out the latitude:
    a_lat=str(a_soup.find_all('meta',property="og:latitude"))
    a_lat=[a_lat.split('"')[1]]
    #Parse out the longitude:
    a_long=str(a_soup.find_all('meta',property="og:longitude"))
    a_long=[a_long.split('"')[1]]
    #Parse out the long description including street address:
    a_name_address=str(a_soup.find_all('meta', property="og:description"))
    a_name_address=a_name_address.split('"')[1]
    a_name_address=[a_name_address.split(str(len(playground_urls)))[0]]
    #Turn the components into a dictionary to make it easier to process later.
    #This could also have been integrated into the former steps but is at least as transparent this way:
    dict={}
    dict['a_name']=a_name
    dict['a_lat']=a_lat
    dict['a_long']=a_long
    dict['a_name_address']=a_name_address
    #return the dictionary:
    return dict
```

#### A function that retrieves a longer playground description and user rating:


```python
'''
The source website includes a space for users to write a longer description of the playground and various notes.
For examples, they sometimes note the most appropriate age range for the equipment available.
Some playgrounds also have a star rating (with 5-star being the best). However, not all are rated.
This data is mostly retrieved for personal use.
'''
def descriptions_and_rating(a_soup): 
    #Sometimes these sections are blank, preparing the result for this:
    a_description=['']
    a_feature=['']
    a_rating=['']
    #Retrieve the site descriptions:
    for heading in a_soup.find_all("h3"):
        #Heading "Beschreibung" has the long description of the site which sometimes includes the best age range:
        if heading.text.strip()=="Beschreibung":
            #Get to the description following the heading:
            a_description=heading.next.next
            #Only retrieve the portion needed:
            pattern = '<div class="description">'
            string = str(a_description)
            repl = ' '
            a_description = re.sub(pattern, repl, string, count=1)
            pattern = '</div>'
            string = str(a_description)
            repl = ' '
            a_description = re.sub(pattern, repl, string, count=1)
            a_description=[a_description]
        ##There were rarely other short notes that were ultimately not used:
        ##These are additional short notes, ex. if the site has shade:
        #elif heading.text.strip()=='Features':
        #    a_feature=[heading.next.next]
        #Sometimes the playground sites also report a user-assigned star rating within the heading "Bewertungen/ Kommentare".
        #So, also retrieving that:
        elif heading.text.strip()=='Bewertungen/ Kommentare':
            #Getting to the data under the heading:
            a_rating=str(heading.next.next.next) 
            #If there isn't a rating, the site encourages the user to add one, we don't need this:
            if a_rating!='Leider wurden noch keine Bewertungen getätigt.' :
                #If there is a rating, parse it out and return it:
                if a_rating.split('"')[3]!='':
                    a_rating=[a_rating.split('"')[3]]
                #If there isn't a rating, return a blank:
                else:
                    a_rating=[''] 
            else:
                a_rating=['']
    #Turn the results into a dicionary for later convenience:
    dict={}
    dict['a_description']=a_description
    dict['a_rating']=a_rating
    #Return the dictionary:
    return dict
```

#### A function that builds a dictionary of German-English playground equipment names:


```python
'''
As the source website is in German, I find it useful to build a German-English dictionary of playground equipment.
The German names of common playground equipment types is the key, and the English correspondence as a value. I am also
using the dictionary in a later step to keep track of the equipment at a playground. So, the dictionary is of form
{German name: [English name, 0]} and the zero is then replaced with a count later.
'''
def equipment_dictionary():
    #Prepare an empty dictionary:
    DE_EN_dictionary=  {} 
    #Make a list of German playground equipment:
    DE_text=["Wasserspiel","Sand","Seilbahn","Spielhaus","Baumhaus","Rutsche","Schaukel","Kletter",
             "Rodelberg","Bolzplatz","Wippe","Basketball","Nestschaukel","Schwingschaukel",
             "Drehscheibe","Karussell","Tischtennis","Trampolin","eisenbahn","traktor","Bagger",
             "Kletterturm","Tunnel","Federbrett","Blancierbretter","Toilette","Fahrradständ"]
    #Make a list of the same equipment in English:
    EN_text=["Water feature","Sandpit","Cable car","Playhouse","tree house","slide","Swing","climbing features",
             "sledding Hill","Football field","seesaw","Basketball","Nest swing","swings",
             "turntable","carousel","table tennis","trampoline","railroad","tractor","Excavator",
             "Climbing tower","tunnel","Spring board","Blancing boards","Toilets",
             "Bicycle stand"]
    #Populating the dicionary via a while loop:
    counter=0
    while counter<len(DE_text):
        #At each iteration, the next German word is used as a key and the next English word is assigned as a value:
        #I also save the keys and values as all lower case for easier searching and matching.
        DE_EN_dictionary[DE_text[counter].lower()] = EN_text[counter].lower()
        counter=counter+1
    #Change the dictionary value to a list and add a second entry of zero for each to be used in the next step:
    #This could also have been done in the former step, but is particularly clear here.
    for key in DE_EN_dictionary:
        DE_EN_dictionary[key]=[DE_EN_dictionary[key],0]
    #Return the resulting dictionary:
    return DE_EN_dictionary
```

#### A pair of functions that retrieve any playground equipment listed:


```python
'''
The source website sometimes makes note of the types of playground equipment available.
The website is great for the intended use, but the html for this area is particularly difficult to retrieve and parse.
So, I retrieve the start of the relevant section and the 1,000 chacters to follow. I then transform this information 
into a string, parse out just the section needed, clean it up a little, and then search through it for the dictionary
keys from the prior function. When the relevant words are found, the playground equipment dictionary also acts as a 
counter and records their existence.
'''
#This function takes a string, searches for key words against the dictionary, and counts when they're found:
def equipment_counter(temp_features, DE_EN_dictionary):  
    #Remove any leading spaces from the string:
    temp_features = temp_features.strip()
    #Make the string all lower case:
    temp_features = temp_features.lower()
    # Split the string into words noting that a lot of different punctuation is used in html:
    words_split = re.findall( r'\w+|[^\s\w]+', temp_features)
    #Loop through the equipment dictionary and the parsed string to search for any words of interest and count them:
    for key in DE_EN_dictionary.keys():
        for word in words_split:
            # Check if the word is in the dictionary:
            if word in DE_EN_dictionary.keys() and word==key:
                #If in the dictionary, add one to the count:
                DE_EN_dictionary[key][1] = DE_EN_dictionary[key][1] + 1
            else:
                #If not in the dictionary, don't add to the count:
                DE_EN_dictionary[key][1]
    #Return the dictionary which now also includes the count of equipment at the playground:
    return DE_EN_dictionary

#This function generally calls the former one plus the equipment dictionary. 
#But first it retrieves the html code that might include equipment:
def playground_equipment_counted(a_soup):
    #As the html is a bit unstructured, retrieve as a string:
    all_string=str(a_soup)
    #Keep the part of the string that might contain the playground equipment:
    #From the heading "Spielplatzgeräte" (playground equipment) plus 1,000 characters.
    temp_features=all_string[all_string.find('Spielplatzgeräte'):all_string.find('Spielplatzgeräte')+1000]
    #Remove more pieces of the string which aren't needed and could cause double-counting:
    pattern = '</h3>.*"/>'
    string = temp_features
    repl = ' '
    temp_features = re.sub(pattern, repl, string, count=1)
    #Next, use the functions equipment_dictionary() and equipment_counter(temp_features, DE_EN_dictionary) to make 
    #counts of the playground equipment available at each site in the dicionary:
    DE_EN_equipment_counted=equipment_counter(temp_features, equipment_dictionary() )
    #Return the playground equipment dictionary, but now with the playground equipment for the site counted:
    return DE_EN_equipment_counted
```

#### Making a dataframe of playground information:
The dataframe will contain the information for all listed playgrounds in the search city's area.


```python
#Make the column names for the resulting dataframe:
a_columns=('a_name', 'a_lat', 'a_long', 'a_name_address', 'a_description',
       'a_rating', 'water feature', 'sandpit', 'cable car', 'playhouse',
       'tree house', 'slide', 'swing', 'climbing features', 'sledding hill',
       'football field', 'seesaw', 'basketball', 'nest swing', 'swings',
       'turntable', 'carousel', 'table tennis', 'trampoline', 'railroad',
       'tractor', 'excavator', 'climbing tower', 'tunnel', 'spring board',
       'blancing boards', 'toilets', 'bicycle stand', 'total equipment')
#Make an empty dataframe with the column headers in which to put the results:
playground_df = pd.DataFrame(columns=a_columns)
#For each playground url in the list from part I, retrieve the playground's details:
for i in range(len(playground_urls)):
    #Assign i's url to be the one retrieved in this loop:
    a_url=playground_urls[i]
    #In-process the playground site's information:
    a_data  = requests.get(a_url).text
    a_soup = BeautifulSoup(a_data,"html5lib")
    #Run the prior functions to process and format the playground's information:
    out1=name_location_data(a_soup)
    out2=descriptions_and_rating(a_soup)
    out3=playground_equipment_counted(a_soup)
    #Convert each function's results into dataframes:
    a_output=pd.DataFrame(out1)
    b_output=pd.DataFrame(out2)
    c_output=pd.DataFrame(out3)
    #The c_output is the dictionary of playground equipment counts.
    #Initially it has two rows. The first row has equipment names which are then used as the column names:
    c_output.columns= c_output.iloc[0].copy()
    #The equipment counts are then retained as the first row and the extra row dropped:
    c_output.iloc[0]=c_output.iloc[1]
    c_output.drop(index=c_output.index[1], axis=0, inplace=True)
    #I then add an extra column which is a sum of the number of equipment available:
    total_equipment=int(c_output[c_output.columns].sum().sum())
    c_output['total equipment']=total_equipment
    #Combine the dataframes for a playground into one output which is a single row with several columns:
    result = pd.concat([a_output, b_output, c_output], axis=1, join='inner')
    #The playground's row is then appended to the dataframe containing all the playground's information:
    playground_df=playground_df.append(result, ignore_index=False, verify_integrity=False, sort=None)
#For each feature count, treat as a 0/1 indicator variable:
feature_list=['water feature', 'sandpit', 'cable car', 'playhouse',
       'tree house', 'slide', 'swing', 'climbing features', 'sledding hill',
       'football field', 'seesaw', 'basketball', 'nest swing', 'swings',
       'turntable', 'carousel', 'table tennis', 'trampoline', 'railroad',
       'tractor', 'excavator', 'climbing tower', 'tunnel', 'spring board',
       'blancing boards', 'toilets', 'bicycle stand']
for feature in feature_list:
    playground_df.loc[playground_df[feature]!=0, feature] = 1
#Show the resulting dataframe:
playground_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a_name</th>
      <th>a_lat</th>
      <th>a_long</th>
      <th>a_name_address</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>53.5926308917772</td>
      <td>9.73169803619385</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Haselweg Wedel</td>
      <td>53.5912910844463</td>
      <td>9.70646917819977</td>
      <td>Schöner Spielplatz mit angegliederter kleiner ...</td>
      <td>Schöner Spielplatz mit angegliederter kleiner...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.5943062279335</td>
      <td>9.71506834030151</td>
      <td>Großer Spielplatz mit viel Wiese. Die Spielger...</td>
      <td>Großer Spielplatz mit viel Wiese. Die Spielge...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>53.5914407324793</td>
      <td>9.70553040504456</td>
      <td>Spielplatz Wasserspielplatz Haus am See in Wed...</td>
      <td>Der Spielplatz macht einen herausragenden Eind...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Mühlenweg Wedel</td>
      <td>53.581066013162</td>
      <td>9.71019208431244</td>
      <td>Schön gestalteter Spielplatz am Mühlenweg.</td>
      <td>Schön gestalteter Spielplatz am Mühlenweg.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Rotdornstraße Wedel</td>
      <td>53.591066930019</td>
      <td>9.68836158514023</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hamburger Yachthafen Wedel</td>
      <td>53.5743968895724</td>
      <td>9.68239367008209</td>
      <td>neuer, riesiger toller spielplatz, muss man hi...</td>
      <td>neuer, riesiger toller spielplatz, muss man h...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Ginsterweg Wedel</td>
      <td>53.5736384685368</td>
      <td>9.71911311149597</td>
      <td>Der Spielplatz ist auf mehrere Ebenen in einem...</td>
      <td>Der Spielplatz ist auf mehrere Ebenen in eine...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hans-Böckler Platz Wedel</td>
      <td>53.5688601987249</td>
      <td>9.71491277217865</td>
      <td>Der Spielplatz ist zur Straße hin mit einem Za...</td>
      <td>Der Spielplatz ist zur Straße hin mit einem Z...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pulverstraße Wedel</td>
      <td>53.5712051224608</td>
      <td>9.71940010786057</td>
      <td>Dieser mittelgroße Spielplatz befindet sich nö...</td>
      <td>Dieser mittelgroße Spielplatz befindet sich n...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Alter Zirkusplatz Wedel</td>
      <td>53.5755961290153</td>
      <td>9.71072316169739</td>
      <td>Versteckter Spielplatz mit schattigen Ecken un...</td>
      <td>Versteckter Spielplatz mit schattigen Ecken u...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Altstadtschule Wedel</td>
      <td>53.582625249582</td>
      <td>9.69926744699478</td>
      <td>Dieser Spielplatz liegt auf dem Schulhof der A...</td>
      <td>Dieser Spielplatz liegt auf dem Schulhof der ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Anne-Frank-Weg Wedel</td>
      <td>53.5889495035841</td>
      <td>9.69376623630524</td>
      <td>Spielplatz mit Matschanlage (also die Ersatzkl...</td>
      <td>Spielplatz mit Matschanlage (also die Ersatzk...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Ansgariusweg Wedel</td>
      <td>53.5871962578912</td>
      <td>9.68622386455536</td>
      <td>An der Zufahrt zum Fährmannssand liegt dieser ...</td>
      <td>An der Zufahrt zum Fährmannssand liegt dieser...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Brombeerweg Wedel</td>
      <td>53.5741194511191</td>
      <td>9.72591519355774</td>
      <td>Schöner kleiner Spielplatz unter Bäumen</td>
      <td>Schöner kleiner Spielplatz unter Bäumen&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Croningstraße Wedel</td>
      <td>53.5819099697564</td>
      <td>9.72337782382965</td>
      <td>Dieser Spielplatz nur von der Croningstraße er...</td>
      <td>Dieser Spielplatz nur von der Croningstraße e...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Gärtnerstraße Wedel</td>
      <td>53.5856072564417</td>
      <td>9.69738721847534</td>
      <td>Dieser Spielplatz wurde seit meinem letzten Be...</td>
      <td>Dieser Spielplatz wurde seit meinem letzten B...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Ernst-Thälmann-Weg Wedel</td>
      <td>53.5881674618245</td>
      <td>9.69368040561676</td>
      <td>Eingeschlossen von Häusern liegt hier ein ruhi...</td>
      <td>Eingeschlossen von Häusern liegt hier ein ruh...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Geesthang Wedel</td>
      <td>53.5897862207119</td>
      <td>9.68335154549268</td>
      <td>Dieser Spielplatz befindet sich am Ende der Ha...</td>
      <td>Dieser Spielplatz befindet sich am Ende der H...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Heinrich-Schacht-Straße Wedel</td>
      <td>53.5800213944949</td>
      <td>9.72487986087799</td>
      <td>Ein größerer Spielplatz. Bemerkenswert neben d...</td>
      <td>Ein größerer Spielplatz. Bemerkenswert neben ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Lindenstraße Wedel</td>
      <td>53.5784133245173</td>
      <td>9.7196630962585</td>
      <td>Ein langezogener Spielplatz mit insgesamt vier...</td>
      <td>Ein langezogener Spielplatz mit insgesamt vie...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pferdekoppel Wedel</td>
      <td>53.5889794348674</td>
      <td>9.7067803144455</td>
      <td>Dieser Spielplatz mit schöner Spielburg liegt ...</td>
      <td>Dieser Spielplatz mit schöner Spielburg liegt...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pinneberger Straße Wedel</td>
      <td>53.5855916527196</td>
      <td>9.70323175191879</td>
      <td>Abgegrenzt vom Obstbaumweg an der Rückseite un...</td>
      <td>Abgegrenzt vom Obstbaumweg an der Rückseite u...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Rosengarten Wedel</td>
      <td>53.5810532740654</td>
      <td>9.70545530319214</td>
      <td>Dieser Spielplatz liegt am Rosengarten, nahe d...</td>
      <td>Dieser Spielplatz liegt am Rosengarten, nahe ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Schwartenseekamp Wedel</td>
      <td>53.5989169842047</td>
      <td>9.73109203352019</td>
      <td>Diesen Spielplatz haben wir noch nicht besucht...</td>
      <td>Diesen Spielplatz haben wir noch nicht besuch...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Strandbad Wedel</td>
      <td>53.5709480505284</td>
      <td>9.69663619995117</td>
      <td>Großer Spielplatz:Wegen des Wassers in der Näh...</td>
      <td>Großer Spielplatz:&lt;br/&gt;Wegen des Wassers in d...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Vogt-Körner Straße Wedel</td>
      <td>53.5751228077681</td>
      <td>9.70517635345459</td>
      <td>Dieser kleine Spielplatz liegt versteckt hinte...</td>
      <td>Dieser kleine Spielplatz liegt versteckt hint...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Wacholderstraße Wedel</td>
      <td>53.5907752727735</td>
      <td>9.69387352466583</td>
      <td>Spielplatz mit Schwerpunkt Sandspiele.</td>
      <td>Spielplatz mit Schwerpunkt Sandspiele.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Kronskamp Wedel</td>
      <td>53.5807953065342</td>
      <td>9.72207427024841</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Appelboomtwiete Ecke Steinberg Wedel</td>
      <td>53.5888347076387</td>
      <td>9.69773345623709</td>
      <td>Diesen Spielplatz haben wir noch nicht besucht...</td>
      <td>Diesen Spielplatz haben wir noch nicht besuch...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Albert-Schweizer Schule Wedel</td>
      <td>53.5717208544489</td>
      <td>9.72274482250214</td>
      <td>Dieser Spielplatz befindet sich auf dem Geländ...</td>
      <td>Dieser Spielplatz befindet sich auf dem Gelän...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Bürgerpark Wedel</td>
      <td>53.5848168731614</td>
      <td>9.69250559806824</td>
      <td>Kleiner Spielplatz im Bürgerpark.</td>
      <td>Kleiner Spielplatz im Bürgerpark.</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Egenbüttelweg Wedel</td>
      <td>53.5911799625823</td>
      <td>9.72104430198669</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Elbstraße Wedel</td>
      <td>53.5699401349262</td>
      <td>9.7113025188446</td>
      <td>Kleiner Spielplatz.</td>
      <td>Kleiner Spielplatz.&lt;br/&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Gerhart-Hauptmann Straße Wedel</td>
      <td>53.591999437962</td>
      <td>9.72143810255561</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hainbuchenweg Wedel</td>
      <td>53.5899384230329</td>
      <td>9.69445219130904</td>
      <td>Spielplatz eher für etwas ältere Kinder.</td>
      <td>Spielplatz eher für etwas ältere Kinder.&lt;br/&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hellgrund Wedel</td>
      <td>53.5669774121414</td>
      <td>9.72199380397797</td>
      <td>Versteckt im Tal in direkter Nähe zum Vattenfa...</td>
      <td>Versteckt im Tal in direkter Nähe zum Vattenf...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Klintkamp Wedel</td>
      <td>53.5919426661751</td>
      <td>9.71341580374904</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Opn Klint Wedel</td>
      <td>53.5873395516784</td>
      <td>9.70869541168213</td>
      <td>Dieser Spielplatz ist zum Teil öffentlch, zum ...</td>
      <td>Dieser Spielplatz ist zum Teil öffentlch, zum...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Reepschlägerstraße Wedel</td>
      <td>53.5863301162117</td>
      <td>9.69326734542847</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Schlehdornweg Wedel</td>
      <td>53.5898009446568</td>
      <td>9.69020962715149</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Von-Suttner Straße Wedel</td>
      <td>53.59188443917404</td>
      <td>9.724164381623268</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Wiedetwiete Wedel</td>
      <td>53.5884908446434</td>
      <td>9.70395349985231</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Autal Wedel</td>
      <td>53.5824138041649</td>
      <td>9.71122829167579</td>
      <td>Sehr kleiner Spielplatz. Gelegen nahe an der W...</td>
      <td>Sehr kleiner Spielplatz. Gelegen nahe an der ...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Im Grund Wedel</td>
      <td>53.575454069497</td>
      <td>9.7262316942215</td>
      <td>Zunächst findet man hier den Bolzplatz, dahint...</td>
      <td>Zunächst findet man hier den Bolzplatz, dahin...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Parnaßstraße Wedel</td>
      <td>53.5695355602878</td>
      <td>9.70255315303802</td>
      <td>Kleiner Spielplatz im Parnaßpark, dem die Graf...</td>
      <td>Kleiner Spielplatz im Parnaßpark, dem die Gra...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Rebhuhnweg Wedel</td>
      <td>53.5933545865536</td>
      <td>9.71821457147598</td>
      <td>Kleiner Spielplatz</td>
      <td>Kleiner Spielplatz&lt;br/&gt;</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Appelboomtwiete Ecke Aastwiete Wedel</td>
      <td>53.59039491694423</td>
      <td>9.69691358503951</td>
      <td>Spielplatz Appelboomtwiete Ecke Aastwiete in W...</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Tinsdaler Weg Wedel</td>
      <td>53.5754046192883</td>
      <td>9.71914261579514</td>
      <td>Spielplatz Tinsdaler Weg in Wedel, Tinsdaler W...</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Theaterstraße Wedel</td>
      <td>53.5822166562335</td>
      <td>9.70818042755127</td>
      <td>Spielplatz Theaterstraße in Wedel in der Theat...</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Heinestraße Wedel</td>
      <td>53.5937489838527</td>
      <td>9.73056882619858</td>
      <td>Spielplatz Heinestraße in Wedel in der Heinest...</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Changing the dataframe's latitudes and longitudes to float type:
playground_df['a_lat']=playground_df['a_lat'].astype(float)
playground_df['a_long']=playground_df['a_long'].astype(float)
#Copying the dataframe. The dataframe built from the website's information has no more changes done to it and now serves
#as source material for the rest of the process:
p_df=playground_df.copy()
```

### Part IC: Visualizing the playground dataset <a name="part1C"></a>

#### Get some starting coordinates for mapping the playgrounds:


```python
#This can certainly be done by taking an average of the playground coordinates or something like that.
#Another way is to use geolocator:
#Searching for the coordinates based on a search for the city in Germany 'search_city':
address = '{}, Germany' .format(search_city)
geolocator = Nominatim(user_agent="playground_explorer")
#Call and retrieve the location data:
location = geolocator.geocode(address)
#Retrieve the latitude:
latitude = location.latitude
#Retrieve the longitude:
longitude = location.longitude
#Display the geocoordinates and the city name:
print('The geograpical coordinate of {} are {}, {}.'.format(address, latitude, longitude))
```

    The geograpical coordinate of Wedel, Germany are 53.5810226, 9.7038772.
    

### Visualize the playground dataset:


```python
#Create a folium map of the city's area using the latitude and longitude values:
map_playgrounds = folium.Map(location=[latitude, longitude], zoom_start=13)
#Add markers to the map for each playground in a loop over the dataframe:
for lat, lng, playground in zip(p_df['a_lat'], p_df['a_long'], p_df['a_name']):
    #Assign the playground name to the marker label:
    label = '{}'.format(playground)
    #Make the label pop up when clicked on:
    label = folium.Popup(label, parse_html=True)
    #Define the folium inputs - lat, long, marker size, color, pop up label data, etc.:
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_playgrounds) #Finally add the point to the map.
#Display the map:
map_playgrounds
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%3Cscript%3EL_PREFER_CANVAS%20%3D%20false%3B%20L_NO_TOUCH%20%3D%20false%3B%20L_DISABLE_3D%20%3D%20false%3B%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawgit.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%20%23map_7678f934b2414bfd9174815457129fa6%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%20%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%20%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_7678f934b2414bfd9174815457129fa6%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20bounds%20%3D%20null%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_7678f934b2414bfd9174815457129fa6%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27map_7678f934b2414bfd9174815457129fa6%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7Bcenter%3A%20%5B53.5810226%2C9.7038772%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2013%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20maxBounds%3A%20bounds%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20layers%3A%20%5B%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20worldCopyJump%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_2cebdb7ad52243ca9b80b16673c3fcbd%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22attribution%22%3A%20null%2C%0A%20%20%22detectRetina%22%3A%20false%2C%0A%20%20%22maxZoom%22%3A%2018%2C%0A%20%20%22minZoom%22%3A%201%2C%0A%20%20%22noWrap%22%3A%20false%2C%0A%20%20%22subdomains%22%3A%20%22abc%22%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_49d6e6b186644edb983e393537e254cf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5926308917772%2C9.73169803619385%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f39f71a8e0614d83aa61f6997d7c2281%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0f586d984f9f44d0bf5b059100900c9a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_0f586d984f9f44d0bf5b059100900c9a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Waldspielplatz%20Moorwegsiedlung%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f39f71a8e0614d83aa61f6997d7c2281.setContent%28html_0f586d984f9f44d0bf5b059100900c9a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_49d6e6b186644edb983e393537e254cf.bindPopup%28popup_f39f71a8e0614d83aa61f6997d7c2281%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_da3ec1544de14e9d8ba6d4215b8b02db%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5912910844463%2C9.70646917819977%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c8797dec02374f9fb0b2ba45bb19ea75%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5b4d709e4e4445d29bce8078488c93b0%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5b4d709e4e4445d29bce8078488c93b0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Haselweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c8797dec02374f9fb0b2ba45bb19ea75.setContent%28html_5b4d709e4e4445d29bce8078488c93b0%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_da3ec1544de14e9d8ba6d4215b8b02db.bindPopup%28popup_c8797dec02374f9fb0b2ba45bb19ea75%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1ef24d7086804721b705493e0effc4e0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5943062279335%2C9.71506834030151%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_ab034ce5586b4cc7a6028ac1dedc67ac%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_96fe44e205cf42f7a456244dc96ad671%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_96fe44e205cf42f7a456244dc96ad671%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Meisenweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_ab034ce5586b4cc7a6028ac1dedc67ac.setContent%28html_96fe44e205cf42f7a456244dc96ad671%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_1ef24d7086804721b705493e0effc4e0.bindPopup%28popup_ab034ce5586b4cc7a6028ac1dedc67ac%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_beb7357ac3dd424193e544ffbcb5a155%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5914407324793%2C9.70553040504456%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2f5f7e84d0514f68b9d2bfa8e8a10d03%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3769e19ab30041319bc21f4705e713f7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_3769e19ab30041319bc21f4705e713f7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wasserspielplatz%20Haus%20am%20See%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2f5f7e84d0514f68b9d2bfa8e8a10d03.setContent%28html_3769e19ab30041319bc21f4705e713f7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_beb7357ac3dd424193e544ffbcb5a155.bindPopup%28popup_2f5f7e84d0514f68b9d2bfa8e8a10d03%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e8d9ff5ab6b043feba605f50f569b5d7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.581066013162%2C9.71019208431244%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_4e033b92977646f88868159414fdc1dd%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_70a985bf9c7445baaf83ff5764f8df31%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_70a985bf9c7445baaf83ff5764f8df31%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20M%C3%BChlenweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_4e033b92977646f88868159414fdc1dd.setContent%28html_70a985bf9c7445baaf83ff5764f8df31%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_e8d9ff5ab6b043feba605f50f569b5d7.bindPopup%28popup_4e033b92977646f88868159414fdc1dd%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3eff9e02b6df4fd786c46d3f2f0ff363%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.591066930019%2C9.68836158514023%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_66ef6b33e8124328b25e641fdcfeb9be%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_faf440298019414181d723c71609ab40%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_faf440298019414181d723c71609ab40%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rotdornstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_66ef6b33e8124328b25e641fdcfeb9be.setContent%28html_faf440298019414181d723c71609ab40%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_3eff9e02b6df4fd786c46d3f2f0ff363.bindPopup%28popup_66ef6b33e8124328b25e641fdcfeb9be%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b5b776819ef741b5874c30e39296107d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5743968895724%2C9.68239367008209%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c8cfbfa616f849138a1f20cbe7895fbd%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1aa50006c0ab4c3ea08dd9bf9b61b7bf%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1aa50006c0ab4c3ea08dd9bf9b61b7bf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hamburger%20Yachthafen%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c8cfbfa616f849138a1f20cbe7895fbd.setContent%28html_1aa50006c0ab4c3ea08dd9bf9b61b7bf%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b5b776819ef741b5874c30e39296107d.bindPopup%28popup_c8cfbfa616f849138a1f20cbe7895fbd%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_47068c72295d4dc2a06532e332986853%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5736384685368%2C9.71911311149597%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d52b1febbbfc4627ad83b11aa3f0bea2%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5755be7006bb41909008f8d20c928dc9%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5755be7006bb41909008f8d20c928dc9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ginsterweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d52b1febbbfc4627ad83b11aa3f0bea2.setContent%28html_5755be7006bb41909008f8d20c928dc9%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_47068c72295d4dc2a06532e332986853.bindPopup%28popup_d52b1febbbfc4627ad83b11aa3f0bea2%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_86363972c9f0413f8124e19bafd637d8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5688601987249%2C9.71491277217865%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_a43749d0c2b549b8b785695ec1148e15%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_27b2a55578704c9fa052dd425e527757%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_27b2a55578704c9fa052dd425e527757%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hans-B%C3%B6ckler%20Platz%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_a43749d0c2b549b8b785695ec1148e15.setContent%28html_27b2a55578704c9fa052dd425e527757%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_86363972c9f0413f8124e19bafd637d8.bindPopup%28popup_a43749d0c2b549b8b785695ec1148e15%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_832f4ce88faa4830bec96871e22c475f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5712051224608%2C9.71940010786057%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_45fd81178db44101a316247f712c7e8c%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_66c6c7447aa64f7a89ae7e48395965c7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_66c6c7447aa64f7a89ae7e48395965c7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pulverstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_45fd81178db44101a316247f712c7e8c.setContent%28html_66c6c7447aa64f7a89ae7e48395965c7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_832f4ce88faa4830bec96871e22c475f.bindPopup%28popup_45fd81178db44101a316247f712c7e8c%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4cbb820454924763847f49f4099474fc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5755961290153%2C9.71072316169739%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_e98887df4ba24cd68a3796448a260875%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_296feed9b2c64bd68fb77ba85a5856c7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_296feed9b2c64bd68fb77ba85a5856c7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Alter%20Zirkusplatz%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_e98887df4ba24cd68a3796448a260875.setContent%28html_296feed9b2c64bd68fb77ba85a5856c7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_4cbb820454924763847f49f4099474fc.bindPopup%28popup_e98887df4ba24cd68a3796448a260875%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c8d6f6a313cc4c52b14892980ad79a96%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.582625249582%2C9.69926744699478%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_9682e889c4e746cbabf18b7692587fc0%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_05617c6a868046e6a6464a11fadbf114%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_05617c6a868046e6a6464a11fadbf114%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Altstadtschule%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_9682e889c4e746cbabf18b7692587fc0.setContent%28html_05617c6a868046e6a6464a11fadbf114%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_c8d6f6a313cc4c52b14892980ad79a96.bindPopup%28popup_9682e889c4e746cbabf18b7692587fc0%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ab92626e3970424e95c611c31a19ece2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5889495035841%2C9.69376623630524%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_4e9f91f143234d17b375f79fed3f5227%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_269918ec001342929dd10cf9717fed9e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_269918ec001342929dd10cf9717fed9e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Anne-Frank-Weg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_4e9f91f143234d17b375f79fed3f5227.setContent%28html_269918ec001342929dd10cf9717fed9e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ab92626e3970424e95c611c31a19ece2.bindPopup%28popup_4e9f91f143234d17b375f79fed3f5227%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_efff4b602ca34c79a2172da66d3cc0b6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5871962578912%2C9.68622386455536%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6bc44aa7219e40fab49a38e0a06923ea%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ab7c46651f5b4cfd833637805cae8255%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_ab7c46651f5b4cfd833637805cae8255%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ansgariusweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6bc44aa7219e40fab49a38e0a06923ea.setContent%28html_ab7c46651f5b4cfd833637805cae8255%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_efff4b602ca34c79a2172da66d3cc0b6.bindPopup%28popup_6bc44aa7219e40fab49a38e0a06923ea%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b3561b1351c440c38d0af6d4f71554b4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5741194511191%2C9.72591519355774%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_88e77a6e0bde44328914d5de9a61fb88%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c39f4e962af04ea2b5b9d8c9832fe4b0%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c39f4e962af04ea2b5b9d8c9832fe4b0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Brombeerweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_88e77a6e0bde44328914d5de9a61fb88.setContent%28html_c39f4e962af04ea2b5b9d8c9832fe4b0%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b3561b1351c440c38d0af6d4f71554b4.bindPopup%28popup_88e77a6e0bde44328914d5de9a61fb88%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_39be2bd600a8477d85d187752373ac9c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5819099697564%2C9.72337782382965%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f8952536621b44c3b8da2ca4b45bb9f2%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c258a92b38634d4aab87479c8bdcc055%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c258a92b38634d4aab87479c8bdcc055%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Croningstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f8952536621b44c3b8da2ca4b45bb9f2.setContent%28html_c258a92b38634d4aab87479c8bdcc055%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_39be2bd600a8477d85d187752373ac9c.bindPopup%28popup_f8952536621b44c3b8da2ca4b45bb9f2%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d148638bcf764620aeb6505c5b0b78ad%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5856072564417%2C9.69738721847534%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_9c79b6cec36548d188846b3f0ca367fc%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5c0d0cee2d90444480a19435b1466379%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5c0d0cee2d90444480a19435b1466379%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20G%C3%A4rtnerstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_9c79b6cec36548d188846b3f0ca367fc.setContent%28html_5c0d0cee2d90444480a19435b1466379%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_d148638bcf764620aeb6505c5b0b78ad.bindPopup%28popup_9c79b6cec36548d188846b3f0ca367fc%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b13615306dae493289445ed0c120c20e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5881674618245%2C9.69368040561676%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6006a29e44654d828f09981e8a126f8b%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c7a93adf0d974d1084d0d6f1255277ea%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c7a93adf0d974d1084d0d6f1255277ea%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ernst-Th%C3%A4lmann-Weg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6006a29e44654d828f09981e8a126f8b.setContent%28html_c7a93adf0d974d1084d0d6f1255277ea%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b13615306dae493289445ed0c120c20e.bindPopup%28popup_6006a29e44654d828f09981e8a126f8b%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a33ca798e05844e8826703bbed9c54fc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5897862207119%2C9.68335154549268%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_3521ba84d7c145969f27bf2d3cceba41%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0cb4d0050fd2467aaaeccc8a6b5f2fb8%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_0cb4d0050fd2467aaaeccc8a6b5f2fb8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Geesthang%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_3521ba84d7c145969f27bf2d3cceba41.setContent%28html_0cb4d0050fd2467aaaeccc8a6b5f2fb8%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a33ca798e05844e8826703bbed9c54fc.bindPopup%28popup_3521ba84d7c145969f27bf2d3cceba41%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a88d2e6fceba42ca800f9b81db402eae%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5800213944949%2C9.72487986087799%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f8f808bab4c5431f80ec486f24a68ff8%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f621896aae79430ea0da28e46c5cad93%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_f621896aae79430ea0da28e46c5cad93%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Heinrich-Schacht-Stra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f8f808bab4c5431f80ec486f24a68ff8.setContent%28html_f621896aae79430ea0da28e46c5cad93%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a88d2e6fceba42ca800f9b81db402eae.bindPopup%28popup_f8f808bab4c5431f80ec486f24a68ff8%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d2ddb0ade6194660b16d7bd08300ec3a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5784133245173%2C9.7196630962585%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d536c8d5be03401da72e92645da14d27%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8f41fbd1e1fe42fea8d0c7c4574f7317%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_8f41fbd1e1fe42fea8d0c7c4574f7317%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Lindenstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d536c8d5be03401da72e92645da14d27.setContent%28html_8f41fbd1e1fe42fea8d0c7c4574f7317%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_d2ddb0ade6194660b16d7bd08300ec3a.bindPopup%28popup_d536c8d5be03401da72e92645da14d27%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_59ed9cf6a2dc41b4b7f9cba97f8eb61b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5889794348674%2C9.7067803144455%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_8d629e0ec21146ab885be4f3e055c182%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ba399817e63a484e82ea0b752d58b488%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_ba399817e63a484e82ea0b752d58b488%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pferdekoppel%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_8d629e0ec21146ab885be4f3e055c182.setContent%28html_ba399817e63a484e82ea0b752d58b488%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_59ed9cf6a2dc41b4b7f9cba97f8eb61b.bindPopup%28popup_8d629e0ec21146ab885be4f3e055c182%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_94665ad271b1412bab7ebfccdcaa2a13%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5855916527196%2C9.70323175191879%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_67fc7879141849aebdc23ae23e695c19%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5e7675f42b45405da1de8cca2ac6f5d9%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5e7675f42b45405da1de8cca2ac6f5d9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pinneberger%20Stra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_67fc7879141849aebdc23ae23e695c19.setContent%28html_5e7675f42b45405da1de8cca2ac6f5d9%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_94665ad271b1412bab7ebfccdcaa2a13.bindPopup%28popup_67fc7879141849aebdc23ae23e695c19%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5b2087a1709e4967a6e5bd1152a9fdd0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5810532740654%2C9.70545530319214%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f950d451362a49aea1e93ee96b15e247%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3dd3802386dc465688a232fa55fb5a8d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_3dd3802386dc465688a232fa55fb5a8d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rosengarten%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f950d451362a49aea1e93ee96b15e247.setContent%28html_3dd3802386dc465688a232fa55fb5a8d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_5b2087a1709e4967a6e5bd1152a9fdd0.bindPopup%28popup_f950d451362a49aea1e93ee96b15e247%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8f60049cf53d432ca71c65d65274f995%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5989169842047%2C9.73109203352019%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_96bd4fb1200b42af8d0b399a473d5315%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5ff19d6dc443405dbb7e4df32930e595%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5ff19d6dc443405dbb7e4df32930e595%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Schwartenseekamp%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_96bd4fb1200b42af8d0b399a473d5315.setContent%28html_5ff19d6dc443405dbb7e4df32930e595%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_8f60049cf53d432ca71c65d65274f995.bindPopup%28popup_96bd4fb1200b42af8d0b399a473d5315%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b0d0fde91abc48cdaf54e312d3288721%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5709480505284%2C9.69663619995117%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_969eb4c005314a2db81b917501517751%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2f583865323c402cb2e01a5012ed176b%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_2f583865323c402cb2e01a5012ed176b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Strandbad%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_969eb4c005314a2db81b917501517751.setContent%28html_2f583865323c402cb2e01a5012ed176b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b0d0fde91abc48cdaf54e312d3288721.bindPopup%28popup_969eb4c005314a2db81b917501517751%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_09f018ca9883498d91ac600636109501%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5751228077681%2C9.70517635345459%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_fea46468c37a45a8aed4a9b5d588ad00%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5d2e22f0449d4521906968d11818f856%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5d2e22f0449d4521906968d11818f856%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Vogt-K%C3%B6rner%20Stra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_fea46468c37a45a8aed4a9b5d588ad00.setContent%28html_5d2e22f0449d4521906968d11818f856%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_09f018ca9883498d91ac600636109501.bindPopup%28popup_fea46468c37a45a8aed4a9b5d588ad00%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_359e5fd808f04fb2a0c8cedab7b2d3b5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5907752727735%2C9.69387352466583%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6e12e2e3a8164541b060c8493cd02d8a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7a747698e9b046d0a6850fb5e71d2836%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_7a747698e9b046d0a6850fb5e71d2836%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wacholderstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6e12e2e3a8164541b060c8493cd02d8a.setContent%28html_7a747698e9b046d0a6850fb5e71d2836%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_359e5fd808f04fb2a0c8cedab7b2d3b5.bindPopup%28popup_6e12e2e3a8164541b060c8493cd02d8a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0f3dbdbd3ea64441a23462825d5b5073%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5807953065342%2C9.72207427024841%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_e7d1c2c8d6194c74bed0050dfd8cd70b%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_be8e35a16bc74354ad6cf13e4b6dbfeb%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_be8e35a16bc74354ad6cf13e4b6dbfeb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Kronskamp%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_e7d1c2c8d6194c74bed0050dfd8cd70b.setContent%28html_be8e35a16bc74354ad6cf13e4b6dbfeb%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0f3dbdbd3ea64441a23462825d5b5073.bindPopup%28popup_e7d1c2c8d6194c74bed0050dfd8cd70b%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_eb2619ac01514fa7bbca825a5af4f7db%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5888347076387%2C9.69773345623709%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6d33495a11fe4aa2b8cd79f1654a423a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6643e7ce82804184b2d74a32232ff519%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_6643e7ce82804184b2d74a32232ff519%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Appelboomtwiete%20Ecke%20Steinberg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6d33495a11fe4aa2b8cd79f1654a423a.setContent%28html_6643e7ce82804184b2d74a32232ff519%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_eb2619ac01514fa7bbca825a5af4f7db.bindPopup%28popup_6d33495a11fe4aa2b8cd79f1654a423a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fe29e1d8756e4d40809a189b3b419ead%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5717208544489%2C9.72274482250214%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0ea87daf4718482bb1a9c30a86756552%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e9f864ebed994f74b4e57225909d384e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_e9f864ebed994f74b4e57225909d384e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Albert-Schweizer%20Schule%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0ea87daf4718482bb1a9c30a86756552.setContent%28html_e9f864ebed994f74b4e57225909d384e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_fe29e1d8756e4d40809a189b3b419ead.bindPopup%28popup_0ea87daf4718482bb1a9c30a86756552%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b7163a5bad6643abb0ab5df49b674d79%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5848168731614%2C9.69250559806824%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_71eb03c51ff7406a9c8a83dea9d60329%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_039d4c59a7da4c24a7a4d4245c78297d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_039d4c59a7da4c24a7a4d4245c78297d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20B%C3%BCrgerpark%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_71eb03c51ff7406a9c8a83dea9d60329.setContent%28html_039d4c59a7da4c24a7a4d4245c78297d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b7163a5bad6643abb0ab5df49b674d79.bindPopup%28popup_71eb03c51ff7406a9c8a83dea9d60329%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5e3d52dff6ba47f09d62d39a812efb62%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5911799625823%2C9.72104430198669%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_77eba1669c0a40f6a1321257f176d3bc%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_592191a024944bb298c211fd1e807c26%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_592191a024944bb298c211fd1e807c26%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Egenb%C3%BCttelweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_77eba1669c0a40f6a1321257f176d3bc.setContent%28html_592191a024944bb298c211fd1e807c26%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_5e3d52dff6ba47f09d62d39a812efb62.bindPopup%28popup_77eba1669c0a40f6a1321257f176d3bc%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ac8aefa5c1604a91989e31cf3642b07a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5699401349262%2C9.7113025188446%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c4ed69631af24f878058eae560c334b6%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a10afc63294f4e0c81bb6e9dbc230788%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_a10afc63294f4e0c81bb6e9dbc230788%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Elbstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c4ed69631af24f878058eae560c334b6.setContent%28html_a10afc63294f4e0c81bb6e9dbc230788%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ac8aefa5c1604a91989e31cf3642b07a.bindPopup%28popup_c4ed69631af24f878058eae560c334b6%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e1573dd5b00949eb906ee24796f2e87b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.591999437962%2C9.72143810255561%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_bad1935b79f74d359402094791266576%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2a3f68645d594eb3991e339a6828d6fc%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_2a3f68645d594eb3991e339a6828d6fc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Gerhart-Hauptmann%20Stra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_bad1935b79f74d359402094791266576.setContent%28html_2a3f68645d594eb3991e339a6828d6fc%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_e1573dd5b00949eb906ee24796f2e87b.bindPopup%28popup_bad1935b79f74d359402094791266576%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_27075bd8e80945979db59a25523f29b2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5899384230329%2C9.69445219130904%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c4e14a9715f54c3aa97b357e9cf7fa39%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_635e393bb3d44c528fbe5a58c8e63556%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_635e393bb3d44c528fbe5a58c8e63556%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hainbuchenweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c4e14a9715f54c3aa97b357e9cf7fa39.setContent%28html_635e393bb3d44c528fbe5a58c8e63556%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_27075bd8e80945979db59a25523f29b2.bindPopup%28popup_c4e14a9715f54c3aa97b357e9cf7fa39%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0ade8d9fdcbf40ae93fc28ea4e2e9862%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5669774121414%2C9.72199380397797%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_5da49213903d44c78b5a3ad6984eddf6%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c0ed9556e5424313a3e94ab28739b64d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c0ed9556e5424313a3e94ab28739b64d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hellgrund%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_5da49213903d44c78b5a3ad6984eddf6.setContent%28html_c0ed9556e5424313a3e94ab28739b64d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0ade8d9fdcbf40ae93fc28ea4e2e9862.bindPopup%28popup_5da49213903d44c78b5a3ad6984eddf6%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_59fcba08a7814a399785c9f431d6fc7b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5919426661751%2C9.71341580374904%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_ca127484f5624eba98de9dbd5fb42e77%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_aecb13e1c2fb4185b17d0b9b74853787%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_aecb13e1c2fb4185b17d0b9b74853787%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Klintkamp%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_ca127484f5624eba98de9dbd5fb42e77.setContent%28html_aecb13e1c2fb4185b17d0b9b74853787%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_59fcba08a7814a399785c9f431d6fc7b.bindPopup%28popup_ca127484f5624eba98de9dbd5fb42e77%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a651fedc68dc4f59bd3ed583f5f9f5bb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5873395516784%2C9.70869541168213%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_8e87a642608841fc98518a9c59e6bece%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_20209b21004c48c9ac7b8a97968288d0%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_20209b21004c48c9ac7b8a97968288d0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Opn%20Klint%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_8e87a642608841fc98518a9c59e6bece.setContent%28html_20209b21004c48c9ac7b8a97968288d0%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a651fedc68dc4f59bd3ed583f5f9f5bb.bindPopup%28popup_8e87a642608841fc98518a9c59e6bece%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_12e853e384c949a98cd9a9e4e16d4b36%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5863301162117%2C9.69326734542847%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_1b0c5f1a3e7346ffa7c9c9866c7fa5e7%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c82760ccb6da4d6fa6e96879ba3be785%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c82760ccb6da4d6fa6e96879ba3be785%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Reepschl%C3%A4gerstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_1b0c5f1a3e7346ffa7c9c9866c7fa5e7.setContent%28html_c82760ccb6da4d6fa6e96879ba3be785%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_12e853e384c949a98cd9a9e4e16d4b36.bindPopup%28popup_1b0c5f1a3e7346ffa7c9c9866c7fa5e7%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1cd11a59394f44ea8c6629747989ecb4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5898009446568%2C9.69020962715149%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_104a0060239f4f57873b8fc35e2fb64a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4d252f16dcd34dbfb9fa65f650e53958%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_4d252f16dcd34dbfb9fa65f650e53958%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Schlehdornweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_104a0060239f4f57873b8fc35e2fb64a.setContent%28html_4d252f16dcd34dbfb9fa65f650e53958%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_1cd11a59394f44ea8c6629747989ecb4.bindPopup%28popup_104a0060239f4f57873b8fc35e2fb64a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3805e34d31d443fc9a53da9b0a64a864%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.59188443917404%2C9.724164381623268%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_1e590ac5b3fa4bada2d4637b67f58af5%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2beaca50ea934cfba2775952e713e2e9%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_2beaca50ea934cfba2775952e713e2e9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Von-Suttner%20Stra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_1e590ac5b3fa4bada2d4637b67f58af5.setContent%28html_2beaca50ea934cfba2775952e713e2e9%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_3805e34d31d443fc9a53da9b0a64a864.bindPopup%28popup_1e590ac5b3fa4bada2d4637b67f58af5%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_adf1f8dbf40244018a2222b80331493a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5884908446434%2C9.70395349985231%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_60375c7aa939430c842abbcb4e722af1%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7d653d2782774a06b469cb3a31ead4a3%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_7d653d2782774a06b469cb3a31ead4a3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wiedetwiete%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_60375c7aa939430c842abbcb4e722af1.setContent%28html_7d653d2782774a06b469cb3a31ead4a3%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_adf1f8dbf40244018a2222b80331493a.bindPopup%28popup_60375c7aa939430c842abbcb4e722af1%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fb4d847dd7284fcea716f394d609d927%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5824138041649%2C9.71122829167579%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_ead7ea8d8496452ab50d2a288244093a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4699e16ff1a747448866cf2f7c60b615%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_4699e16ff1a747448866cf2f7c60b615%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Autal%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_ead7ea8d8496452ab50d2a288244093a.setContent%28html_4699e16ff1a747448866cf2f7c60b615%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_fb4d847dd7284fcea716f394d609d927.bindPopup%28popup_ead7ea8d8496452ab50d2a288244093a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a5e9b4d8b2b941fcad1f58f3d22bc567%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.575454069497%2C9.7262316942215%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_8a1948c0a5bb4e3c8a8943709b7866bf%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c0f41a4b3ccc4d92b138112a2498c83d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c0f41a4b3ccc4d92b138112a2498c83d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Im%20Grund%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_8a1948c0a5bb4e3c8a8943709b7866bf.setContent%28html_c0f41a4b3ccc4d92b138112a2498c83d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a5e9b4d8b2b941fcad1f58f3d22bc567.bindPopup%28popup_8a1948c0a5bb4e3c8a8943709b7866bf%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_515c2aaf48bf413a879a4cc126c2cfed%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5695355602878%2C9.70255315303802%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c3cfb0e4f60543329d0454542149ef17%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_11d0a62c881849f585114044726a5d23%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_11d0a62c881849f585114044726a5d23%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Parna%C3%9Fstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c3cfb0e4f60543329d0454542149ef17.setContent%28html_11d0a62c881849f585114044726a5d23%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_515c2aaf48bf413a879a4cc126c2cfed.bindPopup%28popup_c3cfb0e4f60543329d0454542149ef17%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_16baed08391246ebb703eae6558ea87b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5933545865536%2C9.71821457147598%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d69ad83bdec748369cf0ee201d95df0a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_54d3d33d4ff54a03913c7e7af41b6866%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_54d3d33d4ff54a03913c7e7af41b6866%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rebhuhnweg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d69ad83bdec748369cf0ee201d95df0a.setContent%28html_54d3d33d4ff54a03913c7e7af41b6866%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_16baed08391246ebb703eae6558ea87b.bindPopup%28popup_d69ad83bdec748369cf0ee201d95df0a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b440c840a8c64f78b9f68b3b69d9bd60%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.59039491694423%2C9.69691358503951%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_dfe4efad9dc847318dae964aa9c8f2b5%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0762e14d2ef642629f05737d4348a87e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_0762e14d2ef642629f05737d4348a87e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Appelboomtwiete%20Ecke%20Aastwiete%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_dfe4efad9dc847318dae964aa9c8f2b5.setContent%28html_0762e14d2ef642629f05737d4348a87e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b440c840a8c64f78b9f68b3b69d9bd60.bindPopup%28popup_dfe4efad9dc847318dae964aa9c8f2b5%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cb273f94697f435ba9c8df56194446e2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5754046192883%2C9.71914261579514%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_9933dc8151644f9e8bf4864e0612852f%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1b940329cb35448c98f9ec890602ff25%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1b940329cb35448c98f9ec890602ff25%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Tinsdaler%20Weg%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_9933dc8151644f9e8bf4864e0612852f.setContent%28html_1b940329cb35448c98f9ec890602ff25%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_cb273f94697f435ba9c8df56194446e2.bindPopup%28popup_9933dc8151644f9e8bf4864e0612852f%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_880b53577b5d49c2963534ebbf68481a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5822166562335%2C9.70818042755127%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_7d472a97bc264edda553681d8fb162f5%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e6a5552d779a419c9bf843753349bbfa%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_e6a5552d779a419c9bf843753349bbfa%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Theaterstra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_7d472a97bc264edda553681d8fb162f5.setContent%28html_e6a5552d779a419c9bf843753349bbfa%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_880b53577b5d49c2963534ebbf68481a.bindPopup%28popup_7d472a97bc264edda553681d8fb162f5%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a61faefb47e04867afff360e11eb1b6c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5937489838527%2C9.73056882619858%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_7678f934b2414bfd9174815457129fa6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f4be98906b844cf5a4bfb11a9c0849a5%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8d3a41155fca49a7a9e8be7f14cbacae%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_8d3a41155fca49a7a9e8be7f14cbacae%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Heinestra%C3%9Fe%20Wedel%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f4be98906b844cf5a4bfb11a9c0849a5.setContent%28html_8d3a41155fca49a7a9e8be7f14cbacae%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a61faefb47e04867afff360e11eb1b6c.bindPopup%28popup_f4be98906b844cf5a4bfb11a9c0849a5%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
#It might also be interesting to view the playgrounds in the form of a heat map:
heat_map_playgrounds = folium.Map(location=[latitude, longitude], zoom_start=13)
HeatMap(list(zip(p_df['a_lat'], p_df['a_long'])),radius=35).add_to(heat_map_playgrounds)
heat_map_playgrounds
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%3Cscript%3EL_PREFER_CANVAS%20%3D%20false%3B%20L_NO_TOUCH%20%3D%20false%3B%20L_DISABLE_3D%20%3D%20false%3B%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawgit.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%20%23map_369b53f76f954b0999c876247ec6e4de%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%20%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%20%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//leaflet.github.io/Leaflet.heat/dist/leaflet-heat.js%22%3E%3C/script%3E%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_369b53f76f954b0999c876247ec6e4de%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20bounds%20%3D%20null%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_369b53f76f954b0999c876247ec6e4de%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27map_369b53f76f954b0999c876247ec6e4de%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7Bcenter%3A%20%5B53.5810226%2C9.7038772%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2013%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20maxBounds%3A%20bounds%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20layers%3A%20%5B%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20worldCopyJump%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_238ccccb366f4e84960418da4b823204%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22attribution%22%3A%20null%2C%0A%20%20%22detectRetina%22%3A%20false%2C%0A%20%20%22maxZoom%22%3A%2018%2C%0A%20%20%22minZoom%22%3A%201%2C%0A%20%20%22noWrap%22%3A%20false%2C%0A%20%20%22subdomains%22%3A%20%22abc%22%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_369b53f76f954b0999c876247ec6e4de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20heat_map_755582e0c89a4a85a545cbad2e48dc23%20%3D%20L.heatLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B%5B53.5926308917772%2C%209.73169803619385%5D%2C%20%5B53.5912910844463%2C%209.70646917819977%5D%2C%20%5B53.5943062279335%2C%209.71506834030151%5D%2C%20%5B53.5914407324793%2C%209.70553040504456%5D%2C%20%5B53.581066013162%2C%209.71019208431244%5D%2C%20%5B53.591066930019%2C%209.68836158514023%5D%2C%20%5B53.5743968895724%2C%209.68239367008209%5D%2C%20%5B53.5736384685368%2C%209.71911311149597%5D%2C%20%5B53.5688601987249%2C%209.71491277217865%5D%2C%20%5B53.5712051224608%2C%209.71940010786057%5D%2C%20%5B53.5755961290153%2C%209.71072316169739%5D%2C%20%5B53.582625249582%2C%209.69926744699478%5D%2C%20%5B53.5889495035841%2C%209.69376623630524%5D%2C%20%5B53.5871962578912%2C%209.68622386455536%5D%2C%20%5B53.5741194511191%2C%209.72591519355774%5D%2C%20%5B53.5819099697564%2C%209.72337782382965%5D%2C%20%5B53.5856072564417%2C%209.69738721847534%5D%2C%20%5B53.5881674618245%2C%209.69368040561676%5D%2C%20%5B53.5897862207119%2C%209.68335154549268%5D%2C%20%5B53.5800213944949%2C%209.72487986087799%5D%2C%20%5B53.5784133245173%2C%209.7196630962585%5D%2C%20%5B53.5889794348674%2C%209.7067803144455%5D%2C%20%5B53.5855916527196%2C%209.70323175191879%5D%2C%20%5B53.5810532740654%2C%209.70545530319214%5D%2C%20%5B53.5989169842047%2C%209.73109203352019%5D%2C%20%5B53.5709480505284%2C%209.69663619995117%5D%2C%20%5B53.5751228077681%2C%209.70517635345459%5D%2C%20%5B53.5907752727735%2C%209.69387352466583%5D%2C%20%5B53.5807953065342%2C%209.72207427024841%5D%2C%20%5B53.5888347076387%2C%209.69773345623709%5D%2C%20%5B53.5717208544489%2C%209.72274482250214%5D%2C%20%5B53.5848168731614%2C%209.69250559806824%5D%2C%20%5B53.5911799625823%2C%209.72104430198669%5D%2C%20%5B53.5699401349262%2C%209.7113025188446%5D%2C%20%5B53.591999437962%2C%209.72143810255561%5D%2C%20%5B53.5899384230329%2C%209.69445219130904%5D%2C%20%5B53.5669774121414%2C%209.72199380397797%5D%2C%20%5B53.5919426661751%2C%209.71341580374904%5D%2C%20%5B53.5873395516784%2C%209.70869541168213%5D%2C%20%5B53.5863301162117%2C%209.69326734542847%5D%2C%20%5B53.5898009446568%2C%209.69020962715149%5D%2C%20%5B53.59188443917404%2C%209.724164381623268%5D%2C%20%5B53.5884908446434%2C%209.70395349985231%5D%2C%20%5B53.5824138041649%2C%209.71122829167579%5D%2C%20%5B53.575454069497%2C%209.7262316942215%5D%2C%20%5B53.5695355602878%2C%209.70255315303802%5D%2C%20%5B53.5933545865536%2C%209.71821457147598%5D%2C%20%5B53.59039491694423%2C%209.69691358503951%5D%2C%20%5B53.5754046192883%2C%209.71914261579514%5D%2C%20%5B53.5822166562335%2C%209.70818042755127%5D%2C%20%5B53.5937489838527%2C%209.73056882619858%5D%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20minOpacity%3A%200.5%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20maxZoom%3A%2018%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20max%3A%201.0%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20radius%3A%2035%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20blur%3A%2015%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20gradient%3A%20null%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%29%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20.addTo%28map_369b53f76f954b0999c876247ec6e4de%29%3B%0A%20%20%20%20%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Part ID: Adding Foursquare data <a name="part1D"></a>

### Set up to use Foursquare:

#### Set the Foursquare client information (hidden when sharing):


```python
#Hide
#This square should be hidden when sharing.
#Foursquare ID:
CLIENT_ID = 'HIDDEN'
#Foursquare secret:
CLIENT_SECRET = 'HIDDEN'
```

#### Establish the Foursquare search parameters:


```python
#Foursquare version:
VERSION = '20210520'
#Limit the total number of venues returned for each point:
LIMIT=100
#Search radius in meters around each playground:
radius=500
#The radius could also be a user input at the beginning, after the city name.
```

#### A function that searches Foursquare for venues within a radius around a playground:


```python
'''
This function accesses the Foursquare API, passes the latitudes and longitudes of a set of playgrounds 
(as well as a radius and return limit), and receives back information on the commercial venues within the circle 
defined by the radius around each playground.
'''
def getNearbyVenues(latitudes, longitudes, playgrounds, radius, LIMIT):
    #Create a list to store returned venues in:
    venues_list=[]
    #Run as a loop through the playgrounds in the dataframe when called:
    for lat, lng, playground in zip(latitudes, longitudes, playgrounds):
        print(playground)     
        #Create the API request URL:
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, CLIENT_SECRET, VERSION, lat, lng, radius, LIMIT)    
        #Make the relevant get request:
        results = requests.get(url).json()["response"]['groups'][0]['items']
        #Store the information for each nearby listed venue:
        venues_list.append([(
            playground, lat, lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])
    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    #Define the dataframe column names:
    nearby_venues.columns = ['Playground', 'Playground Latitude', 'Playground Longitude', 
                  'Venue', 'Venue Latitude', 'Venue Longitude', 'Venue Category']
    #Return the venues information:
    return(nearby_venues)
```

#### Call the getNearbyVenues(...) function for the playgrounds dataset:


```python
Playground_venues = getNearbyVenues(playgrounds=p_df['a_name'], 
                                    latitudes=p_df['a_lat'], longitudes=p_df['a_long'], 
                                    radius=radius, LIMIT=LIMIT)
#The function will also print which playgrounds have been called within the function.
```

     Spielplatz Waldspielplatz Moorwegsiedlung Wedel
     Spielplatz Haselweg Wedel
     Spielplatz Meisenweg Wedel
     Spielplatz Wasserspielplatz Haus am See Wedel
     Spielplatz Mühlenweg Wedel
     Spielplatz Rotdornstraße Wedel
     Spielplatz Hamburger Yachthafen Wedel
     Spielplatz Ginsterweg Wedel
     Spielplatz Hans-Böckler Platz Wedel
     Spielplatz Pulverstraße Wedel
     Spielplatz Alter Zirkusplatz Wedel
     Spielplatz Altstadtschule Wedel
     Spielplatz Anne-Frank-Weg Wedel
     Spielplatz Ansgariusweg Wedel
     Spielplatz Brombeerweg Wedel
     Spielplatz Croningstraße Wedel
     Spielplatz Gärtnerstraße Wedel
     Spielplatz Ernst-Thälmann-Weg Wedel
     Spielplatz Geesthang Wedel
     Spielplatz Heinrich-Schacht-Straße Wedel
     Spielplatz Lindenstraße Wedel
     Spielplatz Pferdekoppel Wedel
     Spielplatz Pinneberger Straße Wedel
     Spielplatz Rosengarten Wedel
     Spielplatz Schwartenseekamp Wedel
     Spielplatz Strandbad Wedel
     Spielplatz Vogt-Körner Straße Wedel
     Spielplatz Wacholderstraße Wedel
     Spielplatz Kronskamp Wedel
     Spielplatz Appelboomtwiete Ecke Steinberg Wedel
     Spielplatz Albert-Schweizer Schule Wedel
     Spielplatz Bürgerpark Wedel
     Spielplatz Egenbüttelweg Wedel
     Spielplatz Elbstraße Wedel
     Spielplatz Gerhart-Hauptmann Straße Wedel
     Spielplatz Hainbuchenweg Wedel
     Spielplatz Hellgrund Wedel
     Spielplatz Klintkamp Wedel
     Spielplatz Opn Klint Wedel
     Spielplatz Reepschlägerstraße Wedel
     Spielplatz Schlehdornweg Wedel
     Spielplatz Von-Suttner Straße Wedel
     Spielplatz Wiedetwiete Wedel
     Spielplatz Autal Wedel
     Spielplatz Im Grund Wedel
     Spielplatz Parnaßstraße Wedel
     Spielplatz Rebhuhnweg Wedel
     Spielplatz Appelboomtwiete Ecke Aastwiete Wedel
     Spielplatz Tinsdaler Weg Wedel
     Spielplatz Theaterstraße Wedel
     Spielplatz Heinestraße Wedel
    

#### Check the playgrounds-venues data:


```python
#Check the output size and format:
print(Playground_venues.shape)
#Take a look at the first few rows of the resulting dataframe:
Playground_venues.head(10)
```

    (308, 7)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>Playground Latitude</th>
      <th>Playground Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>53.592631</td>
      <td>9.731698</td>
      <td>Waldspielplatz</td>
      <td>53.592623</td>
      <td>9.731623</td>
      <td>Playground</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>53.592631</td>
      <td>9.731698</td>
      <td>Hackradt Bäcker</td>
      <td>53.590744</td>
      <td>9.725168</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spielplatz Haselweg Wedel</td>
      <td>53.591291</td>
      <td>9.706469</td>
      <td>ALDI NORD</td>
      <td>53.592820</td>
      <td>9.710503</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>9.715068</td>
      <td>ALDI NORD</td>
      <td>53.592820</td>
      <td>9.710503</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>9.715068</td>
      <td>Spielplatz</td>
      <td>53.592523</td>
      <td>9.710625</td>
      <td>Playground</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>9.715068</td>
      <td>Schokoengel</td>
      <td>53.590827</td>
      <td>9.715937</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>9.715068</td>
      <td>elBistro Wedel</td>
      <td>53.590727</td>
      <td>9.715602</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>9.715068</td>
      <td>Spielplatz</td>
      <td>53.592500</td>
      <td>9.720890</td>
      <td>Playground</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>53.591441</td>
      <td>9.705530</td>
      <td>ALDI NORD</td>
      <td>53.592820</td>
      <td>9.710503</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>53.591441</td>
      <td>9.705530</td>
      <td>Steinberghalle Wedel</td>
      <td>53.589295</td>
      <td>9.699597</td>
      <td>College Gym</td>
    </tr>
  </tbody>
</table>
</div>



#### Check which categories are returned and for which playgrounds:


```python
#Find out how many categories of venues have been returned:
print('There are {} uniques categories.'.format(len(Playground_venues['Venue Category'].unique())))
#Get a list of the venue categories:
print(Playground_venues['Venue Category'].unique())
#Get a list of the counts of venues around each playground:
#Note these include the number of other playgrounds ('Spielplatz') nearby.
Playground_venues.groupby('Playground')['Playground'].count()
```

    There are 54 uniques categories.
    ['Playground' 'Bakery' 'Supermarket' 'Café' 'Mexican Restaurant'
     'College Gym' 'Thai Restaurant' 'Italian Restaurant' 'Drugstore' 'Gym'
     'Doner Restaurant' 'Shopping Mall' 'Fast Food Restaurant' 'Garden Center'
     'Seafood Restaurant' 'Harbor / Marina' 'Boat or Ferry' 'Garden'
     'Photography Studio' 'Taverna' 'Bus Stop' 'Beach' 'Turkish Restaurant'
     'Clothing Store' 'Bank' 'Optical Shop' 'Pub' 'Pool' 'Steakhouse'
     'German Restaurant' 'Hotel' 'Trattoria/Osteria' 'Sculpture Garden'
     'Restaurant' 'Museum' 'Theater' 'Insurance Office' 'Tea Room'
     'Food & Drink Shop' 'Arts & Crafts Store' 'Sandwich Place' 'Nightclub'
     'French Restaurant' 'Gym / Fitness Center' 'Furniture / Home Store'
     'Electronics Store' 'Pet Store' 'Asian Restaurant' 'Plaza' 'Spa'
     'Beach Bar' 'Pier' 'Soccer Field' 'Sushi Restaurant']
    




    Playground
     Spielplatz Albert-Schweizer Schule Wedel            5
     Spielplatz Alter Zirkusplatz Wedel                 12
     Spielplatz Altstadtschule Wedel                    20
     Spielplatz Anne-Frank-Weg Wedel                     4
     Spielplatz Ansgariusweg Wedel                       4
     Spielplatz Appelboomtwiete Ecke Aastwiete Wedel     3
     Spielplatz Appelboomtwiete Ecke Steinberg Wedel     3
     Spielplatz Autal Wedel                              6
     Spielplatz Brombeerweg Wedel                        4
     Spielplatz Bürgerpark Wedel                         8
     Spielplatz Croningstraße Wedel                     16
     Spielplatz Egenbüttelweg Wedel                      5
     Spielplatz Elbstraße Wedel                          5
     Spielplatz Ernst-Thälmann-Weg Wedel                 4
     Spielplatz Geesthang Wedel                          1
     Spielplatz Gerhart-Hauptmann Straße Wedel           5
     Spielplatz Ginsterweg Wedel                         5
     Spielplatz Gärtnerstraße Wedel                     15
     Spielplatz Hainbuchenweg Wedel                      4
     Spielplatz Hamburger Yachthafen Wedel               4
     Spielplatz Hans-Böckler Platz Wedel                 4
     Spielplatz Haselweg Wedel                           1
     Spielplatz Heinestraße Wedel                        3
     Spielplatz Heinrich-Schacht-Straße Wedel           11
     Spielplatz Hellgrund Wedel                          1
     Spielplatz Im Grund Wedel                           4
     Spielplatz Klintkamp Wedel                          6
     Spielplatz Kronskamp Wedel                         11
     Spielplatz Lindenstraße Wedel                       4
     Spielplatz Meisenweg Wedel                          5
     Spielplatz Mühlenweg Wedel                         10
     Spielplatz Parnaßstraße Wedel                       8
     Spielplatz Pferdekoppel Wedel                       2
     Spielplatz Pinneberger Straße Wedel                14
     Spielplatz Pulverstraße Wedel                       4
     Spielplatz Rebhuhnweg Wedel                         4
     Spielplatz Reepschlägerstraße Wedel                 8
     Spielplatz Rosengarten Wedel                       21
     Spielplatz Rotdornstraße Wedel                      1
     Spielplatz Schlehdornweg Wedel                      4
     Spielplatz Schwartenseekamp Wedel                   2
     Spielplatz Strandbad Wedel                          9
     Spielplatz Theaterstraße Wedel                     13
     Spielplatz Tinsdaler Weg Wedel                      3
     Spielplatz Vogt-Körner Straße Wedel                 9
     Spielplatz Von-Suttner Straße Wedel                 4
     Spielplatz Wacholderstraße Wedel                    4
     Spielplatz Waldspielplatz Moorwegsiedlung Wedel     2
     Spielplatz Wasserspielplatz Haus am See Wedel       2
     Spielplatz Wiedetwiete Wedel                        1
    Name: Playground, dtype: int64



#### Prepare to analyze the data using dummy variable encoding:


```python
#Use one hot encoding on the venues data as a new dataframe:
Playgrounds_onehot = pd.get_dummies(Playground_venues[['Venue Category']], prefix="", prefix_sep="")
#Add playground names to the dataframe:
Playgrounds_onehot = Playgrounds_onehot.drop(['Playground'], axis=1)
Playgrounds_onehot.insert(0, 'Playground', Playground_venues['Playground'])
#Get the shape of the resulting dataframe and explore the first few rows:
print(Playgrounds_onehot.shape)
Playgrounds_onehot.head(10)
```

    (308, 54)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Beach</th>
      <th>Beach Bar</th>
      <th>Boat or Ferry</th>
      <th>Bus Stop</th>
      <th>Café</th>
      <th>Clothing Store</th>
      <th>College Gym</th>
      <th>Doner Restaurant</th>
      <th>Drugstore</th>
      <th>Electronics Store</th>
      <th>Fast Food Restaurant</th>
      <th>Food &amp; Drink Shop</th>
      <th>French Restaurant</th>
      <th>Furniture / Home Store</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>German Restaurant</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Hotel</th>
      <th>Insurance Office</th>
      <th>Italian Restaurant</th>
      <th>Mexican Restaurant</th>
      <th>Museum</th>
      <th>Nightclub</th>
      <th>Optical Shop</th>
      <th>Pet Store</th>
      <th>Photography Studio</th>
      <th>Pier</th>
      <th>Plaza</th>
      <th>Pool</th>
      <th>Pub</th>
      <th>Restaurant</th>
      <th>Sandwich Place</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shopping Mall</th>
      <th>Soccer Field</th>
      <th>Spa</th>
      <th>Steakhouse</th>
      <th>Supermarket</th>
      <th>Sushi Restaurant</th>
      <th>Taverna</th>
      <th>Tea Room</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Trattoria/Osteria</th>
      <th>Turkish Restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spielplatz Haselweg Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#For the purposes of clustering, normalize these by the mean:
Playgrounds_grouped = Playgrounds_onehot.groupby('Playground').mean().reset_index()
#Again review the dataframe:
print(Playgrounds_grouped.shape)
Playgrounds_grouped.head(10)
```

    (50, 54)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Beach</th>
      <th>Beach Bar</th>
      <th>Boat or Ferry</th>
      <th>Bus Stop</th>
      <th>Café</th>
      <th>Clothing Store</th>
      <th>College Gym</th>
      <th>Doner Restaurant</th>
      <th>Drugstore</th>
      <th>Electronics Store</th>
      <th>Fast Food Restaurant</th>
      <th>Food &amp; Drink Shop</th>
      <th>French Restaurant</th>
      <th>Furniture / Home Store</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>German Restaurant</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Hotel</th>
      <th>Insurance Office</th>
      <th>Italian Restaurant</th>
      <th>Mexican Restaurant</th>
      <th>Museum</th>
      <th>Nightclub</th>
      <th>Optical Shop</th>
      <th>Pet Store</th>
      <th>Photography Studio</th>
      <th>Pier</th>
      <th>Plaza</th>
      <th>Pool</th>
      <th>Pub</th>
      <th>Restaurant</th>
      <th>Sandwich Place</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shopping Mall</th>
      <th>Soccer Field</th>
      <th>Spa</th>
      <th>Steakhouse</th>
      <th>Supermarket</th>
      <th>Sushi Restaurant</th>
      <th>Taverna</th>
      <th>Tea Room</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Trattoria/Osteria</th>
      <th>Turkish Restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Albert-Schweizer Schule Wedel</td>
      <td>0.20</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spielplatz Alter Zirkusplatz Wedel</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.083333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spielplatz Altstadtschule Wedel</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.05</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.100</td>
      <td>0.000000</td>
      <td>0.100</td>
      <td>0.0</td>
      <td>0.050</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>0.050</td>
      <td>0.05</td>
      <td>0.0</td>
      <td>0.050</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.050</td>
      <td>0.05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spielplatz Anne-Frank-Weg Wedel</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.250000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Spielplatz Ansgariusweg Wedel</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spielplatz Appelboomtwiete Ecke Aastwiete Wedel</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.333333</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spielplatz Appelboomtwiete Ecke Steinberg Wedel</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spielplatz Autal Wedel</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Spielplatz Brombeerweg Wedel</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Spielplatz Bürgerpark Wedel</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.125</td>
      <td>0.000000</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.125</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.125000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Part II: Methodology and results<a name="part2"></a>

### Part IIA: Exploring the top five venue types for each playground<a name="part2A"></a>


```python
#Note we want the top 5:
num_top_venues = 5
#Run as a loop of each playground:
for location in Playgrounds_grouped['Playground']:
    print("----"+location+"----")
    #Set up to retrieve the data for a location:
    temp = Playgrounds_grouped[Playgrounds_grouped['Playground'] == location].T.reset_index()
    #Set the resulting data columns:
    temp.columns = ['venue','freq']
    #Load the relevant data into the columns:
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    #Sort and return the results:
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```

    ---- Spielplatz Albert-Schweizer Schule Wedel----
                     venue  freq
    0  Arts & Crafts Store   0.2
    1   Photography Studio   0.2
    2             Bus Stop   0.2
    3          Supermarket   0.2
    4               Garden   0.2
    
    
    ---- Spielplatz Alter Zirkusplatz Wedel----
                    venue  freq
    0       Shopping Mall  0.17
    1         Supermarket  0.17
    2  Turkish Restaurant  0.08
    3            Bus Stop  0.08
    4             Taverna  0.08
    
    
    ---- Spielplatz Altstadtschule Wedel----
                    venue  freq
    0  Italian Restaurant  0.10
    1               Hotel  0.10
    2    Doner Restaurant  0.05
    3    Sculpture Garden  0.05
    4          Restaurant  0.05
    
    
    ---- Spielplatz Anne-Frank-Weg Wedel----
                     venue  freq
    0     Insurance Office  0.25
    1          Supermarket  0.25
    2          College Gym  0.25
    3        Garden Center  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Ansgariusweg Wedel----
                     venue  freq
    0               Bakery  0.25
    1             Tea Room  0.25
    2          Supermarket  0.25
    3        Garden Center  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Appelboomtwiete Ecke Aastwiete Wedel----
                     venue  freq
    0     Insurance Office  0.33
    1             Tea Room  0.33
    2          College Gym  0.33
    3  Arts & Crafts Store  0.00
    4     Sculpture Garden  0.00
    
    
    ---- Spielplatz Appelboomtwiete Ecke Steinberg Wedel----
                     venue  freq
    0             Tea Room  0.33
    1          College Gym  0.33
    2  Arts & Crafts Store  0.00
    3     Sculpture Garden  0.00
    4            Nightclub  0.00
    
    
    ---- Spielplatz Autal Wedel----
                  venue  freq
    0   Thai Restaurant  0.17
    1          Bus Stop  0.17
    2  Sushi Restaurant  0.17
    3        Steakhouse  0.17
    4  Doner Restaurant  0.17
    
    
    ---- Spielplatz Brombeerweg Wedel----
                     venue  freq
    0  Arts & Crafts Store  0.25
    1    Food & Drink Shop  0.25
    2             Bus Stop  0.25
    3                 Café  0.25
    4         Optical Shop  0.00
    
    
    ---- Spielplatz Bürgerpark Wedel----
                    venue  freq
    0  Italian Restaurant  0.12
    1         Supermarket  0.12
    2               Hotel  0.12
    3                 Pub  0.12
    4    Sculpture Garden  0.12
    
    
    ---- Spielplatz Croningstraße Wedel----
                        venue  freq
    0             Supermarket  0.19
    1    Fast Food Restaurant  0.12
    2    Gym / Fitness Center  0.06
    3  Furniture / Home Store  0.06
    4       French Restaurant  0.06
    
    
    ---- Spielplatz Egenbüttelweg Wedel----
                    venue  freq
    0  Mexican Restaurant   0.2
    1              Bakery   0.2
    2                Café   0.2
    3          Restaurant   0.2
    4  Seafood Restaurant   0.0
    
    
    ---- Spielplatz Elbstraße Wedel----
                     venue  freq
    0             Bus Stop   0.4
    1               Bakery   0.2
    2                Beach   0.2
    3          Supermarket   0.2
    4  Arts & Crafts Store   0.0
    
    
    ---- Spielplatz Ernst-Thälmann-Weg Wedel----
                     venue  freq
    0     Insurance Office  0.25
    1          Supermarket  0.25
    2          College Gym  0.25
    3        Garden Center  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Geesthang Wedel----
                     venue  freq
    0        Garden Center   1.0
    1  Arts & Crafts Store   0.0
    2     Sculpture Garden   0.0
    3            Nightclub   0.0
    4         Optical Shop   0.0
    
    
    ---- Spielplatz Gerhart-Hauptmann Straße Wedel----
                    venue  freq
    0  Mexican Restaurant   0.2
    1              Bakery   0.2
    2                Café   0.2
    3          Restaurant   0.2
    4  Seafood Restaurant   0.0
    
    
    ---- Spielplatz Ginsterweg Wedel----
                    venue  freq
    0  Photography Studio   0.2
    1             Taverna   0.2
    2            Bus Stop   0.2
    3         Supermarket   0.2
    4              Garden   0.2
    
    
    ---- Spielplatz Gärtnerstraße Wedel----
                    venue  freq
    0               Hotel  0.13
    1  Italian Restaurant  0.07
    2         Supermarket  0.07
    3   German Restaurant  0.07
    4                 Pub  0.07
    
    
    ---- Spielplatz Hainbuchenweg Wedel----
                     venue  freq
    0     Insurance Office  0.25
    1             Tea Room  0.25
    2          College Gym  0.25
    3        Garden Center  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Hamburger Yachthafen Wedel----
                     venue  freq
    0        Boat or Ferry  0.50
    1   Seafood Restaurant  0.25
    2      Harbor / Marina  0.25
    3  Arts & Crafts Store  0.00
    4         Optical Shop  0.00
    
    
    ---- Spielplatz Hans-Böckler Platz Wedel----
                     venue  freq
    0               Bakery  0.25
    1                Beach  0.25
    2             Bus Stop  0.25
    3          Supermarket  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Haselweg Wedel----
                     venue  freq
    0          Supermarket   1.0
    1  Arts & Crafts Store   0.0
    2     Sculpture Garden   0.0
    3            Nightclub   0.0
    4         Optical Shop   0.0
    
    
    ---- Spielplatz Heinestraße Wedel----
                     venue  freq
    0               Bakery  0.33
    1           Restaurant  0.33
    2  Arts & Crafts Store  0.00
    3   Seafood Restaurant  0.00
    4            Nightclub  0.00
    
    
    ---- Spielplatz Heinrich-Schacht-Straße Wedel----
                      venue  freq
    0           Supermarket  0.27
    1  Fast Food Restaurant  0.18
    2     French Restaurant  0.09
    3                Bakery  0.09
    4               Taverna  0.09
    
    
    ---- Spielplatz Hellgrund Wedel----
                     venue  freq
    0                Beach   1.0
    1  Arts & Crafts Store   0.0
    2   Seafood Restaurant   0.0
    3            Nightclub   0.0
    4         Optical Shop   0.0
    
    
    ---- Spielplatz Im Grund Wedel----
                     venue  freq
    0    Food & Drink Shop  0.25
    1             Bus Stop  0.25
    2                 Café  0.25
    3           Restaurant  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Klintkamp Wedel----
                     venue  freq
    0                 Café  0.17
    1          Supermarket  0.17
    2   Mexican Restaurant  0.17
    3  Arts & Crafts Store  0.00
    4     Sculpture Garden  0.00
    
    
    ---- Spielplatz Kronskamp Wedel----
                      venue  freq
    0           Supermarket  0.27
    1  Fast Food Restaurant  0.18
    2     French Restaurant  0.09
    3                Bakery  0.09
    4               Taverna  0.09
    
    
    ---- Spielplatz Lindenstraße Wedel----
                     venue  freq
    0              Taverna  0.50
    1             Bus Stop  0.25
    2    French Restaurant  0.25
    3  Arts & Crafts Store  0.00
    4     Sculpture Garden  0.00
    
    
    ---- Spielplatz Meisenweg Wedel----
                     venue  freq
    0                 Café   0.2
    1          Supermarket   0.2
    2   Mexican Restaurant   0.2
    3  Arts & Crafts Store   0.0
    4     Sculpture Garden   0.0
    
    
    ---- Spielplatz Mühlenweg Wedel----
                    venue  freq
    0              Bakery   0.2
    1           Drugstore   0.2
    2  Italian Restaurant   0.1
    3    Doner Restaurant   0.1
    4                 Gym   0.1
    
    
    ---- Spielplatz Parnaßstraße Wedel----
                    venue  freq
    0  Seafood Restaurant  0.25
    1               Beach  0.12
    2           Beach Bar  0.12
    3                Pier  0.12
    4            Bus Stop  0.12
    
    
    ---- Spielplatz Pferdekoppel Wedel----
                     venue  freq
    0          Supermarket   0.5
    1          College Gym   0.5
    2  Arts & Crafts Store   0.0
    3     Sculpture Garden   0.0
    4            Nightclub   0.0
    
    
    ---- Spielplatz Pinneberger Straße Wedel----
                    venue  freq
    0  Italian Restaurant  0.14
    1               Hotel  0.14
    2         College Gym  0.07
    3          Restaurant  0.07
    4   German Restaurant  0.07
    
    
    ---- Spielplatz Pulverstraße Wedel----
                     venue  freq
    0               Bakery  0.25
    1             Bus Stop  0.25
    2          Supermarket  0.25
    3               Garden  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Rebhuhnweg Wedel----
                     venue  freq
    0                 Café  0.25
    1           Restaurant  0.25
    2   Mexican Restaurant  0.25
    3  Arts & Crafts Store  0.00
    4     Sculpture Garden  0.00
    
    
    ---- Spielplatz Reepschlägerstraße Wedel----
                    venue  freq
    0  Italian Restaurant  0.12
    1               Hotel  0.12
    2         Supermarket  0.12
    3          Steakhouse  0.12
    4       Garden Center  0.12
    
    
    ---- Spielplatz Rosengarten Wedel----
                    venue  freq
    0              Bakery  0.10
    1                Café  0.10
    2           Drugstore  0.10
    3  Italian Restaurant  0.05
    4   German Restaurant  0.05
    
    
    ---- Spielplatz Rotdornstraße Wedel----
                     venue  freq
    0        Garden Center   1.0
    1  Arts & Crafts Store   0.0
    2     Sculpture Garden   0.0
    3            Nightclub   0.0
    4         Optical Shop   0.0
    
    
    ---- Spielplatz Schlehdornweg Wedel----
                     venue  freq
    0               Bakery  0.25
    1             Tea Room  0.25
    2          Supermarket  0.25
    3        Garden Center  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Schwartenseekamp Wedel----
                     venue  freq
    0                Plaza   0.5
    1                  Spa   0.5
    2  Arts & Crafts Store   0.0
    3     Sculpture Garden   0.0
    4            Nightclub   0.0
    
    
    ---- Spielplatz Strandbad Wedel----
                    venue  freq
    0  Seafood Restaurant  0.33
    1               Beach  0.11
    2           Beach Bar  0.11
    3               Hotel  0.11
    4     Harbor / Marina  0.11
    
    
    ---- Spielplatz Theaterstraße Wedel----
                    venue  freq
    0           Drugstore  0.15
    1  Italian Restaurant  0.08
    2                Café  0.08
    3                 Gym  0.08
    4          Restaurant  0.08
    
    
    ---- Spielplatz Tinsdaler Weg Wedel----
                     venue  freq
    0   Photography Studio  0.33
    1              Taverna  0.33
    2               Garden  0.33
    3  Arts & Crafts Store  0.00
    4     Sculpture Garden  0.00
    
    
    ---- Spielplatz Vogt-Körner Straße Wedel----
                    venue  freq
    0         Supermarket  0.33
    1  Turkish Restaurant  0.11
    2        Optical Shop  0.11
    3                Café  0.11
    4      Clothing Store  0.11
    
    
    ---- Spielplatz Von-Suttner Straße Wedel----
                     venue  freq
    0               Bakery  0.25
    1           Restaurant  0.25
    2  Arts & Crafts Store  0.00
    3   Seafood Restaurant  0.00
    4            Nightclub  0.00
    
    
    ---- Spielplatz Wacholderstraße Wedel----
                     venue  freq
    0     Insurance Office  0.25
    1             Tea Room  0.25
    2          College Gym  0.25
    3        Garden Center  0.25
    4  Arts & Crafts Store  0.00
    
    
    ---- Spielplatz Waldspielplatz Moorwegsiedlung Wedel----
                     venue  freq
    0               Bakery   0.5
    1  Arts & Crafts Store   0.0
    2   Seafood Restaurant   0.0
    3            Nightclub   0.0
    4         Optical Shop   0.0
    
    
    ---- Spielplatz Wasserspielplatz Haus am See Wedel----
                     venue  freq
    0          Supermarket   0.5
    1          College Gym   0.5
    2  Arts & Crafts Store   0.0
    3     Sculpture Garden   0.0
    4            Nightclub   0.0
    
    
    ---- Spielplatz Wiedetwiete Wedel----
                     venue  freq
    0          College Gym   1.0
    1  Arts & Crafts Store   0.0
    2   Seafood Restaurant   0.0
    3            Nightclub   0.0
    4         Optical Shop   0.0
    
    
    

#### A function that sorts and returns the most common venues:


```python
def return_most_common_venues(row, num_top_venues):
    #Note the playground row's categories:
    row_categories = row.iloc[1:]
    #Sort the data by most common venue category:
    row_categories_sorted = row_categories.sort_values(ascending=False)
    #Return the sorted data:
    return row_categories_sorted.index.values[0:num_top_venues]
```

#### Display the playgrounds with their frequency of categories returned: 


```python
#Return the top ten most common venue categories. 
#However, top ten might be a bit much, something like top five would probably be pretty informative.
num_top_venues = 10
#Use these endings to make the numbers more readable:
indicators = ['st', 'nd', 'rd']
#Create columns by the number of top venues:
columns = ['Playground']
#Generate the column names to assign the venues to:
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))
#Create a new dataframe results and assign the column labels just generated to it:
playgrounds_venues_sorted = pd.DataFrame(columns=columns)
#Assign data to the new dataframe from above to start from:
playgrounds_venues_sorted['Playground'] = Playgrounds_grouped['Playground']
#Sort through the data for each playground, get the name of the most common venue, and put into the dataframe cell:
for ind in np.arange(Playgrounds_grouped.shape[0]):
    #Note calling of the prior function to get the ranked venues:
    playgrounds_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Playgrounds_grouped.iloc[ind, :], num_top_venues)
#Return the dataframe with the most common venues for each playground noted:
playgrounds_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Albert-Schweizer Schule Wedel</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Garden</td>
      <td>Supermarket</td>
      <td>Photography Studio</td>
      <td>Bus Stop</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spielplatz Alter Zirkusplatz Wedel</td>
      <td>Supermarket</td>
      <td>Shopping Mall</td>
      <td>Turkish Restaurant</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Bank</td>
      <td>Taverna</td>
      <td>Optical Shop</td>
      <td>Clothing Store</td>
      <td>Bus Stop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spielplatz Altstadtschule Wedel</td>
      <td>Italian Restaurant</td>
      <td>Hotel</td>
      <td>Sculpture Garden</td>
      <td>Trattoria/Osteria</td>
      <td>Museum</td>
      <td>Fast Food Restaurant</td>
      <td>Drugstore</td>
      <td>Doner Restaurant</td>
      <td>Pool</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spielplatz Anne-Frank-Weg Wedel</td>
      <td>Garden Center</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Spielplatz Ansgariusweg Wedel</td>
      <td>Bakery</td>
      <td>Tea Room</td>
      <td>Supermarket</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
    </tr>
  </tbody>
</table>
</div>



### Part IIB: Clustering and mapping based on venues <a name="part2B"></a>

#### Fit the k-means algorithm to the playground surrounding venues normalized data:


```python
#Set the number of clusters (five seems to work fine):
kclusters = 5
#Drop the name and set the dataset to use:
Playgrounds_grouped_clustering = Playgrounds_grouped.drop('Playground', 1)
#Run/fit the k-means clustering algorithm:
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Playgrounds_grouped_clustering)
#Check the cluster labels generated:
kmeans.labels_[0:len(Playgrounds_grouped)]
```




    array([3, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 0, 4, 1, 3, 1, 0, 1, 3, 0,
           1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 1, 4, 0, 1, 1, 1, 1,
           1, 1, 0, 1, 0, 0])



#### Create a new dataframe that includes the cluster labels as well as the top 10 venues around each playground:


```python
#Insert the cluster labels into the sorted/ranked venues dataframe:
playgrounds_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
#Copy the playground dataframe and rename the playground column to use in merging:
Playgrounds_merged=p_df.copy().rename(columns={"a_name":"Playground"})
#Merge the playgrounds_grouped and playgrounds_df (p_df) dataframes so as to add latitude/longitudes:
final_df = Playgrounds_merged.join(playgrounds_venues_sorted.set_index('Playground'), on='Playground')
#Note the dataframe's dimensions:
print(final_df.shape)
#Move the cluster labels to the first column:
first_column = final_df.pop('Cluster Labels')
final_df.insert(0, 'Cluster Labels', first_column)
#There are a couple of remote playgrounds without venues listed 
#These will instead be added separately to the map:
exception_df=final_df[final_df['Cluster Labels'].isna()]
final_df = final_df[final_df['Cluster Labels'].notna()]
#The couple outlier playgrounds then caused the cluster labels to be float, so changing back to int:
final_df['Cluster Labels']=final_df['Cluster Labels'].astype(int)
#Display the resulting dataframe:
final_df.head(60)
```

    (51, 45)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster Labels</th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_long</th>
      <th>a_name_address</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>53.592631</td>
      <td>9.731698</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>Bakery</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Haselweg Wedel</td>
      <td>53.591291</td>
      <td>9.706469</td>
      <td>Schöner Spielplatz mit angegliederter kleiner ...</td>
      <td>Schöner Spielplatz mit angegliederter kleiner...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Supermarket</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>9.715068</td>
      <td>Großer Spielplatz mit viel Wiese. Die Spielger...</td>
      <td>Großer Spielplatz mit viel Wiese. Die Spielge...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Mexican Restaurant</td>
      <td>Supermarket</td>
      <td>Café</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>53.591441</td>
      <td>9.705530</td>
      <td>Spielplatz Wasserspielplatz Haus am See in Wed...</td>
      <td>Der Spielplatz macht einen herausragenden Eind...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Mühlenweg Wedel</td>
      <td>53.581066</td>
      <td>9.710192</td>
      <td>Schön gestalteter Spielplatz am Mühlenweg.</td>
      <td>Schön gestalteter Spielplatz am Mühlenweg.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Bakery</td>
      <td>Drugstore</td>
      <td>Italian Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Gym</td>
      <td>Shopping Mall</td>
      <td>Fast Food Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>Spielplatz Rotdornstraße Wedel</td>
      <td>53.591067</td>
      <td>9.688362</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Insurance Office</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Hamburger Yachthafen Wedel</td>
      <td>53.574397</td>
      <td>9.682394</td>
      <td>neuer, riesiger toller spielplatz, muss man hi...</td>
      <td>neuer, riesiger toller spielplatz, muss man h...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Boat or Ferry</td>
      <td>Harbor / Marina</td>
      <td>Seafood Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Spielplatz Ginsterweg Wedel</td>
      <td>53.573638</td>
      <td>9.719113</td>
      <td>Der Spielplatz ist auf mehrere Ebenen in einem...</td>
      <td>Der Spielplatz ist auf mehrere Ebenen in eine...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>Taverna</td>
      <td>Supermarket</td>
      <td>Garden</td>
      <td>Bus Stop</td>
      <td>Photography Studio</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Spielplatz Hans-Böckler Platz Wedel</td>
      <td>53.568860</td>
      <td>9.714913</td>
      <td>Der Spielplatz ist zur Straße hin mit einem Za...</td>
      <td>Der Spielplatz ist zur Straße hin mit einem Z...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Bakery</td>
      <td>Beach</td>
      <td>Supermarket</td>
      <td>Bus Stop</td>
      <td>Turkish Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Spielplatz Pulverstraße Wedel</td>
      <td>53.571205</td>
      <td>9.719400</td>
      <td>Dieser mittelgroße Spielplatz befindet sich nö...</td>
      <td>Dieser mittelgroße Spielplatz befindet sich n...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Bakery</td>
      <td>Supermarket</td>
      <td>Bus Stop</td>
      <td>Garden</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Alter Zirkusplatz Wedel</td>
      <td>53.575596</td>
      <td>9.710723</td>
      <td>Versteckter Spielplatz mit schattigen Ecken un...</td>
      <td>Versteckter Spielplatz mit schattigen Ecken u...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Supermarket</td>
      <td>Shopping Mall</td>
      <td>Turkish Restaurant</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Bank</td>
      <td>Taverna</td>
      <td>Optical Shop</td>
      <td>Clothing Store</td>
      <td>Bus Stop</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Altstadtschule Wedel</td>
      <td>53.582625</td>
      <td>9.699267</td>
      <td>Dieser Spielplatz liegt auf dem Schulhof der A...</td>
      <td>Dieser Spielplatz liegt auf dem Schulhof der ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Italian Restaurant</td>
      <td>Hotel</td>
      <td>Sculpture Garden</td>
      <td>Trattoria/Osteria</td>
      <td>Museum</td>
      <td>Fast Food Restaurant</td>
      <td>Drugstore</td>
      <td>Doner Restaurant</td>
      <td>Pool</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Anne-Frank-Weg Wedel</td>
      <td>53.588950</td>
      <td>9.693766</td>
      <td>Spielplatz mit Matschanlage (also die Ersatzkl...</td>
      <td>Spielplatz mit Matschanlage (also die Ersatzk...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Garden Center</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Ansgariusweg Wedel</td>
      <td>53.587196</td>
      <td>9.686224</td>
      <td>An der Zufahrt zum Fährmannssand liegt dieser ...</td>
      <td>An der Zufahrt zum Fährmannssand liegt dieser...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Bakery</td>
      <td>Tea Room</td>
      <td>Supermarket</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Brombeerweg Wedel</td>
      <td>53.574119</td>
      <td>9.725915</td>
      <td>Schöner kleiner Spielplatz unter Bäumen</td>
      <td>Schöner kleiner Spielplatz unter Bäumen&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Bus Stop</td>
      <td>Café</td>
      <td>Food &amp; Drink Shop</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Croningstraße Wedel</td>
      <td>53.581910</td>
      <td>9.723378</td>
      <td>Dieser Spielplatz nur von der Croningstraße er...</td>
      <td>Dieser Spielplatz nur von der Croningstraße e...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Supermarket</td>
      <td>Fast Food Restaurant</td>
      <td>Furniture / Home Store</td>
      <td>Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Pet Store</td>
      <td>French Restaurant</td>
      <td>Nightclub</td>
      <td>Sandwich Place</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Gärtnerstraße Wedel</td>
      <td>53.585607</td>
      <td>9.697387</td>
      <td>Dieser Spielplatz wurde seit meinem letzten Be...</td>
      <td>Dieser Spielplatz wurde seit meinem letzten B...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Hotel</td>
      <td>Italian Restaurant</td>
      <td>Sculpture Garden</td>
      <td>Trattoria/Osteria</td>
      <td>Museum</td>
      <td>College Gym</td>
      <td>Pub</td>
      <td>German Restaurant</td>
      <td>Café</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Ernst-Thälmann-Weg Wedel</td>
      <td>53.588167</td>
      <td>9.693680</td>
      <td>Eingeschlossen von Häusern liegt hier ein ruhi...</td>
      <td>Eingeschlossen von Häusern liegt hier ein ruh...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Garden Center</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>Spielplatz Geesthang Wedel</td>
      <td>53.589786</td>
      <td>9.683352</td>
      <td>Dieser Spielplatz befindet sich am Ende der Ha...</td>
      <td>Dieser Spielplatz befindet sich am Ende der H...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Insurance Office</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Heinrich-Schacht-Straße Wedel</td>
      <td>53.580021</td>
      <td>9.724880</td>
      <td>Ein größerer Spielplatz. Bemerkenswert neben d...</td>
      <td>Ein größerer Spielplatz. Bemerkenswert neben ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Supermarket</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>French Restaurant</td>
      <td>Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Taverna</td>
      <td>Bakery</td>
      <td>Beach</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Lindenstraße Wedel</td>
      <td>53.578413</td>
      <td>9.719663</td>
      <td>Ein langezogener Spielplatz mit insgesamt vier...</td>
      <td>Ein langezogener Spielplatz mit insgesamt vie...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Taverna</td>
      <td>Bus Stop</td>
      <td>French Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Pferdekoppel Wedel</td>
      <td>53.588979</td>
      <td>9.706780</td>
      <td>Dieser Spielplatz mit schöner Spielburg liegt ...</td>
      <td>Dieser Spielplatz mit schöner Spielburg liegt...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Pinneberger Straße Wedel</td>
      <td>53.585592</td>
      <td>9.703232</td>
      <td>Abgegrenzt vom Obstbaumweg an der Rückseite un...</td>
      <td>Abgegrenzt vom Obstbaumweg an der Rückseite u...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Italian Restaurant</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>German Restaurant</td>
      <td>Trattoria/Osteria</td>
      <td>Doner Restaurant</td>
      <td>Pub</td>
      <td>College Gym</td>
      <td>Sculpture Garden</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Rosengarten Wedel</td>
      <td>53.581053</td>
      <td>9.705455</td>
      <td>Dieser Spielplatz liegt am Rosengarten, nahe d...</td>
      <td>Dieser Spielplatz liegt am Rosengarten, nahe ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Drugstore</td>
      <td>Turkish Restaurant</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Doner Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>German Restaurant</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Schwartenseekamp Wedel</td>
      <td>53.598917</td>
      <td>9.731092</td>
      <td>Diesen Spielplatz haben wir noch nicht besucht...</td>
      <td>Diesen Spielplatz haben wir noch nicht besuch...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Spa</td>
      <td>Plaza</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Strandbad Wedel</td>
      <td>53.570948</td>
      <td>9.696636</td>
      <td>Großer Spielplatz:Wegen des Wassers in der Näh...</td>
      <td>Großer Spielplatz:&lt;br/&gt;Wegen des Wassers in d...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Seafood Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Beach</td>
      <td>Beach Bar</td>
      <td>Soccer Field</td>
      <td>Pier</td>
      <td>Hotel</td>
      <td>Fast Food Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Vogt-Körner Straße Wedel</td>
      <td>53.575123</td>
      <td>9.705176</td>
      <td>Dieser kleine Spielplatz liegt versteckt hinte...</td>
      <td>Dieser kleine Spielplatz liegt versteckt hint...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Supermarket</td>
      <td>Turkish Restaurant</td>
      <td>Optical Shop</td>
      <td>Café</td>
      <td>Shopping Mall</td>
      <td>Clothing Store</td>
      <td>Drugstore</td>
      <td>Fast Food Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Wacholderstraße Wedel</td>
      <td>53.590775</td>
      <td>9.693874</td>
      <td>Spielplatz mit Schwerpunkt Sandspiele.</td>
      <td>Spielplatz mit Schwerpunkt Sandspiele.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Tea Room</td>
      <td>Garden Center</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Kronskamp Wedel</td>
      <td>53.580795</td>
      <td>9.722074</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Supermarket</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>French Restaurant</td>
      <td>Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Taverna</td>
      <td>Bakery</td>
      <td>Beach</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Appelboomtwiete Ecke Steinberg Wedel</td>
      <td>53.588835</td>
      <td>9.697733</td>
      <td>Diesen Spielplatz haben wir noch nicht besucht...</td>
      <td>Diesen Spielplatz haben wir noch nicht besuch...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Tea Room</td>
      <td>College Gym</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Spielplatz Albert-Schweizer Schule Wedel</td>
      <td>53.571721</td>
      <td>9.722745</td>
      <td>Dieser Spielplatz befindet sich auf dem Geländ...</td>
      <td>Dieser Spielplatz befindet sich auf dem Gelän...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Garden</td>
      <td>Supermarket</td>
      <td>Photography Studio</td>
      <td>Bus Stop</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Bürgerpark Wedel</td>
      <td>53.584817</td>
      <td>9.692506</td>
      <td>Kleiner Spielplatz im Bürgerpark.</td>
      <td>Kleiner Spielplatz im Bürgerpark.</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Italian Restaurant</td>
      <td>Supermarket</td>
      <td>Sculpture Garden</td>
      <td>Museum</td>
      <td>Steakhouse</td>
      <td>Hotel</td>
      <td>Theater</td>
      <td>Pub</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Egenbüttelweg Wedel</td>
      <td>53.591180</td>
      <td>9.721044</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Turkish Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Spielplatz Elbstraße Wedel</td>
      <td>53.569940</td>
      <td>9.711303</td>
      <td>Kleiner Spielplatz.</td>
      <td>Kleiner Spielplatz.&lt;br/&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Bus Stop</td>
      <td>Bakery</td>
      <td>Beach</td>
      <td>Supermarket</td>
      <td>Turkish Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Gerhart-Hauptmann Straße Wedel</td>
      <td>53.591999</td>
      <td>9.721438</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Turkish Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Hainbuchenweg Wedel</td>
      <td>53.589938</td>
      <td>9.694452</td>
      <td>Spielplatz eher für etwas ältere Kinder.</td>
      <td>Spielplatz eher für etwas ältere Kinder.&lt;br/&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Tea Room</td>
      <td>Garden Center</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Spielplatz Hellgrund Wedel</td>
      <td>53.566977</td>
      <td>9.721994</td>
      <td>Versteckt im Tal in direkter Nähe zum Vattenfa...</td>
      <td>Versteckt im Tal in direkter Nähe zum Vattenf...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Beach</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Klintkamp Wedel</td>
      <td>53.591943</td>
      <td>9.713416</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Mexican Restaurant</td>
      <td>Supermarket</td>
      <td>Café</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Reepschlägerstraße Wedel</td>
      <td>53.586330</td>
      <td>9.693267</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Italian Restaurant</td>
      <td>Supermarket</td>
      <td>Garden Center</td>
      <td>Museum</td>
      <td>Pub</td>
      <td>Sculpture Garden</td>
      <td>Steakhouse</td>
      <td>Hotel</td>
      <td>Beach Bar</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Schlehdornweg Wedel</td>
      <td>53.589801</td>
      <td>9.690210</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Bakery</td>
      <td>Tea Room</td>
      <td>Supermarket</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Von-Suttner Straße Wedel</td>
      <td>53.591884</td>
      <td>9.724164</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Bakery</td>
      <td>Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Wiedetwiete Wedel</td>
      <td>53.588491</td>
      <td>9.703953</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>College Gym</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Autal Wedel</td>
      <td>53.582414</td>
      <td>9.711228</td>
      <td>Sehr kleiner Spielplatz. Gelegen nahe an der W...</td>
      <td>Sehr kleiner Spielplatz. Gelegen nahe an der ...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Thai Restaurant</td>
      <td>Gym</td>
      <td>Sushi Restaurant</td>
      <td>Steakhouse</td>
      <td>Bus Stop</td>
      <td>Doner Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Im Grund Wedel</td>
      <td>53.575454</td>
      <td>9.726232</td>
      <td>Zunächst findet man hier den Bolzplatz, dahint...</td>
      <td>Zunächst findet man hier den Bolzplatz, dahin...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Restaurant</td>
      <td>Bus Stop</td>
      <td>Café</td>
      <td>Food &amp; Drink Shop</td>
      <td>Turkish Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Parnaßstraße Wedel</td>
      <td>53.569536</td>
      <td>9.702553</td>
      <td>Kleiner Spielplatz im Parnaßpark, dem die Graf...</td>
      <td>Kleiner Spielplatz im Parnaßpark, dem die Gra...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Seafood Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Beach</td>
      <td>Beach Bar</td>
      <td>Bus Stop</td>
      <td>Pier</td>
      <td>Hotel</td>
      <td>Fast Food Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Rebhuhnweg Wedel</td>
      <td>53.593355</td>
      <td>9.718215</td>
      <td>Kleiner Spielplatz</td>
      <td>Kleiner Spielplatz&lt;br/&gt;</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Mexican Restaurant</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Appelboomtwiete Ecke Aastwiete Wedel</td>
      <td>53.590395</td>
      <td>9.696914</td>
      <td>Spielplatz Appelboomtwiete Ecke Aastwiete in W...</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Tea Room</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Tinsdaler Weg Wedel</td>
      <td>53.575405</td>
      <td>9.719143</td>
      <td>Spielplatz Tinsdaler Weg in Wedel, Tinsdaler W...</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Taverna</td>
      <td>Garden</td>
      <td>Photography Studio</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Theaterstraße Wedel</td>
      <td>53.582217</td>
      <td>9.708180</td>
      <td>Spielplatz Theaterstraße in Wedel in der Theat...</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Drugstore</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Trattoria/Osteria</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Taverna</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Heinestraße Wedel</td>
      <td>53.593749</td>
      <td>9.730569</td>
      <td>Spielplatz Heinestraße in Wedel in der Heinest...</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Bakery</td>
      <td>Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
  </tbody>
</table>
</div>



#### Visualize the clustering result:


```python
#Create another folium map:
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=13)
#Set color scheme for the clusters, doing so in a way that is flexible to the number of clusters::
#Get a set of evenly spaced values to use in color selection, then use in choosing color values:
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
#Access the colors available using hex codes:
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
#Add markers to the map for each playground based on cluster:
for lat, lon, poi, cluster in zip(final_df['a_lat'], final_df['a_long'], final_df['Playground'], final_df['Cluster Labels']):
    #Label the markers based on cluster and playground name, set the cluster parameters and add to the map:
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters) 
#Add the playground points for the outliers that didn't have venues nearby to use in clustering:    
for lat, lon, poi in zip(exception_df['a_lat'], exception_df['a_long'], exception_df['Playground']):
    #Label these exceptions as outliers and map in the color black:
    label = folium.Popup(str(poi) + ' *Note not clustered - no venues listed', parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color='#000000',
        fill=True,
        fill_color='#000000',
        fill_opacity=0.7).add_to(map_clusters)
#Display the map with the clustered and non-clustered playgrounds:
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%3Cscript%3EL_PREFER_CANVAS%20%3D%20false%3B%20L_NO_TOUCH%20%3D%20false%3B%20L_DISABLE_3D%20%3D%20false%3B%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawgit.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%20%23map_86885ca849f34cd7bfe1c34b28696a3b%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%20%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%20%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_86885ca849f34cd7bfe1c34b28696a3b%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20bounds%20%3D%20null%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_86885ca849f34cd7bfe1c34b28696a3b%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27map_86885ca849f34cd7bfe1c34b28696a3b%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7Bcenter%3A%20%5B53.5810226%2C9.7038772%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2013%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20maxBounds%3A%20bounds%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20layers%3A%20%5B%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20worldCopyJump%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_5c4b5c244ac14a909d65c250715b7fd6%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22attribution%22%3A%20null%2C%0A%20%20%22detectRetina%22%3A%20false%2C%0A%20%20%22maxZoom%22%3A%2018%2C%0A%20%20%22minZoom%22%3A%201%2C%0A%20%20%22noWrap%22%3A%20false%2C%0A%20%20%22subdomains%22%3A%20%22abc%22%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1b5158b901ab4819a6878ffcd7d626ac%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5926308917772%2C9.73169803619385%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_1c4ed469b673419ba60ae4ddb9938759%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5a3d5d9dd5f84b5ea75d74f9c7c08912%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5a3d5d9dd5f84b5ea75d74f9c7c08912%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Waldspielplatz%20Moorwegsiedlung%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_1c4ed469b673419ba60ae4ddb9938759.setContent%28html_5a3d5d9dd5f84b5ea75d74f9c7c08912%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_1b5158b901ab4819a6878ffcd7d626ac.bindPopup%28popup_1c4ed469b673419ba60ae4ddb9938759%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_652bf7a7f6a344b9aa88488a995ecdbf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5912910844463%2C9.70646917819977%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d368f15641e74396974e41662a5af490%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_73ce6a1f753e4cebbc0ed6b0ed431606%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_73ce6a1f753e4cebbc0ed6b0ed431606%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Haselweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d368f15641e74396974e41662a5af490.setContent%28html_73ce6a1f753e4cebbc0ed6b0ed431606%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_652bf7a7f6a344b9aa88488a995ecdbf.bindPopup%28popup_d368f15641e74396974e41662a5af490%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_640db79d3bbe48329e265d9e20ede078%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5943062279335%2C9.71506834030151%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_9fa2d8db4ee344848bfbee66b3ee3006%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a9060fefec314a46852fee1a6672faaf%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_a9060fefec314a46852fee1a6672faaf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Meisenweg%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_9fa2d8db4ee344848bfbee66b3ee3006.setContent%28html_a9060fefec314a46852fee1a6672faaf%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_640db79d3bbe48329e265d9e20ede078.bindPopup%28popup_9fa2d8db4ee344848bfbee66b3ee3006%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4f7cd76204274eee94868bd0a16092e7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5914407324793%2C9.70553040504456%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_63af9dbf684c423bb4ccb1521de9070a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_39b76b3aa38c4228bd99f4dcbc9cd4e9%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_39b76b3aa38c4228bd99f4dcbc9cd4e9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wasserspielplatz%20Haus%20am%20See%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_63af9dbf684c423bb4ccb1521de9070a.setContent%28html_39b76b3aa38c4228bd99f4dcbc9cd4e9%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_4f7cd76204274eee94868bd0a16092e7.bindPopup%28popup_63af9dbf684c423bb4ccb1521de9070a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_88469ba2b9ae49ab8f3cedeb8b53ce87%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.581066013162%2C9.71019208431244%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2503efa8a6294ebbad1aea8e6ae7ea31%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d6487390968a4ec58ec3216521bd3c4d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_d6487390968a4ec58ec3216521bd3c4d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20M%C3%BChlenweg%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2503efa8a6294ebbad1aea8e6ae7ea31.setContent%28html_d6487390968a4ec58ec3216521bd3c4d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_88469ba2b9ae49ab8f3cedeb8b53ce87.bindPopup%28popup_2503efa8a6294ebbad1aea8e6ae7ea31%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0bfe130337fa4bc78842d00f1f3b9b19%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.591066930019%2C9.68836158514023%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0fbcba5da8184414b70f1250500eb054%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c3873af7cce747608e98fbd77ed85176%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c3873af7cce747608e98fbd77ed85176%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rotdornstra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0fbcba5da8184414b70f1250500eb054.setContent%28html_c3873af7cce747608e98fbd77ed85176%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0bfe130337fa4bc78842d00f1f3b9b19.bindPopup%28popup_0fbcba5da8184414b70f1250500eb054%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6dc03f9504f048a08be79c30aa341d82%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5743968895724%2C9.68239367008209%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_28f7accecc944a6cb50ca3683b1619b2%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dbb4cd4a398e48cb808422eb6282bb7a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_dbb4cd4a398e48cb808422eb6282bb7a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hamburger%20Yachthafen%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_28f7accecc944a6cb50ca3683b1619b2.setContent%28html_dbb4cd4a398e48cb808422eb6282bb7a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_6dc03f9504f048a08be79c30aa341d82.bindPopup%28popup_28f7accecc944a6cb50ca3683b1619b2%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3e888e0edc2b4a5aa8a5776d8d2e1e86%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5736384685368%2C9.71911311149597%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_36369251d00146a89d0d378ee202e838%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_139a49967011493eb8ad8537474d9308%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_139a49967011493eb8ad8537474d9308%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ginsterweg%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_36369251d00146a89d0d378ee202e838.setContent%28html_139a49967011493eb8ad8537474d9308%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_3e888e0edc2b4a5aa8a5776d8d2e1e86.bindPopup%28popup_36369251d00146a89d0d378ee202e838%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5dcd2b1a587448f59b1f612bad57a18c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5688601987249%2C9.71491277217865%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2274fb8f6afc489f875ee8bcef553b8d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fee30365392a46aaa02b7c302e8d9222%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_fee30365392a46aaa02b7c302e8d9222%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hans-B%C3%B6ckler%20Platz%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2274fb8f6afc489f875ee8bcef553b8d.setContent%28html_fee30365392a46aaa02b7c302e8d9222%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_5dcd2b1a587448f59b1f612bad57a18c.bindPopup%28popup_2274fb8f6afc489f875ee8bcef553b8d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_50cb41219116420793a32411670a08e8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5712051224608%2C9.71940010786057%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_73a880f57ccd493b9312b5c32585fb9a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a35df61042924c48a5ffb158d7016d6f%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_a35df61042924c48a5ffb158d7016d6f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pulverstra%C3%9Fe%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_73a880f57ccd493b9312b5c32585fb9a.setContent%28html_a35df61042924c48a5ffb158d7016d6f%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_50cb41219116420793a32411670a08e8.bindPopup%28popup_73a880f57ccd493b9312b5c32585fb9a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e4d4ebcf21264eabaa38cd39c7a31320%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5755961290153%2C9.71072316169739%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_e931b12f1ee94ca89c2afe656c60b176%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d8b8b44e86334960a59460a3e20867e6%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_d8b8b44e86334960a59460a3e20867e6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Alter%20Zirkusplatz%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_e931b12f1ee94ca89c2afe656c60b176.setContent%28html_d8b8b44e86334960a59460a3e20867e6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_e4d4ebcf21264eabaa38cd39c7a31320.bindPopup%28popup_e931b12f1ee94ca89c2afe656c60b176%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d45135c04a4344a6a3e33e1c008d9cec%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.582625249582%2C9.69926744699478%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2618492df85b4d37982dc0daf803cf6d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b18b297b7db44d379f9db885080ba71e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b18b297b7db44d379f9db885080ba71e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Altstadtschule%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2618492df85b4d37982dc0daf803cf6d.setContent%28html_b18b297b7db44d379f9db885080ba71e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_d45135c04a4344a6a3e33e1c008d9cec.bindPopup%28popup_2618492df85b4d37982dc0daf803cf6d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3d14675b258e4d7d9837610469d6480f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5889495035841%2C9.69376623630524%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_a0669cbc2be146599113ae139956c884%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_474eb9cc72c8484184d29f3ec86107c2%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_474eb9cc72c8484184d29f3ec86107c2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Anne-Frank-Weg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_a0669cbc2be146599113ae139956c884.setContent%28html_474eb9cc72c8484184d29f3ec86107c2%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_3d14675b258e4d7d9837610469d6480f.bindPopup%28popup_a0669cbc2be146599113ae139956c884%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0bb54ffc64df4d6489cdc01afb645cb5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5871962578912%2C9.68622386455536%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d587cf692ee24953a7178bf0ebbe4da7%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e2457f3caebd40dfb0440e250f76d95d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_e2457f3caebd40dfb0440e250f76d95d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ansgariusweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d587cf692ee24953a7178bf0ebbe4da7.setContent%28html_e2457f3caebd40dfb0440e250f76d95d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0bb54ffc64df4d6489cdc01afb645cb5.bindPopup%28popup_d587cf692ee24953a7178bf0ebbe4da7%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_197053b611954952a0def9327c59a3c0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5741194511191%2C9.72591519355774%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6bb2621ec7f74d409aba287905445c6d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5d849c1fe9054bb0afdc6dddc145ebbb%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5d849c1fe9054bb0afdc6dddc145ebbb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Brombeerweg%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6bb2621ec7f74d409aba287905445c6d.setContent%28html_5d849c1fe9054bb0afdc6dddc145ebbb%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_197053b611954952a0def9327c59a3c0.bindPopup%28popup_6bb2621ec7f74d409aba287905445c6d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a7238301546446fd8f750dd7d17e7557%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5819099697564%2C9.72337782382965%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_ccb72c198c1e4fec91c38e5b87e8a71c%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8fd1ccbbd9064279809b0c8b22c36676%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_8fd1ccbbd9064279809b0c8b22c36676%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Croningstra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_ccb72c198c1e4fec91c38e5b87e8a71c.setContent%28html_8fd1ccbbd9064279809b0c8b22c36676%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a7238301546446fd8f750dd7d17e7557.bindPopup%28popup_ccb72c198c1e4fec91c38e5b87e8a71c%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_846752d2a2894db68520798c2abce345%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5856072564417%2C9.69738721847534%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_bec245b302bc4ae880c3d70195ef469a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1ca2abe872574718b9f97bb7d6cb2732%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1ca2abe872574718b9f97bb7d6cb2732%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20G%C3%A4rtnerstra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_bec245b302bc4ae880c3d70195ef469a.setContent%28html_1ca2abe872574718b9f97bb7d6cb2732%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_846752d2a2894db68520798c2abce345.bindPopup%28popup_bec245b302bc4ae880c3d70195ef469a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cefa495ea5154d698d74d0e49c7c4cf2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5881674618245%2C9.69368040561676%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0700a52b38e34a48b8294a2814082cfc%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_09509e5c34dd4bc4a7a41afe22b0200d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_09509e5c34dd4bc4a7a41afe22b0200d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ernst-Th%C3%A4lmann-Weg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0700a52b38e34a48b8294a2814082cfc.setContent%28html_09509e5c34dd4bc4a7a41afe22b0200d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_cefa495ea5154d698d74d0e49c7c4cf2.bindPopup%28popup_0700a52b38e34a48b8294a2814082cfc%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_56e71526ffc84572800b05a137c0b382%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5897862207119%2C9.68335154549268%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c1c04e18bec44ca886db4b8753120da5%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_983317fb488b46a2a8efa47128e94f9a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_983317fb488b46a2a8efa47128e94f9a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Geesthang%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c1c04e18bec44ca886db4b8753120da5.setContent%28html_983317fb488b46a2a8efa47128e94f9a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_56e71526ffc84572800b05a137c0b382.bindPopup%28popup_c1c04e18bec44ca886db4b8753120da5%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_97295e4073174a2fb1c4e9f06bf9ef7d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5800213944949%2C9.72487986087799%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_778537ae6c6349d8a60ed0f349a1255d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b10f7b0ae9864ce086dd3b7cf1d2d681%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b10f7b0ae9864ce086dd3b7cf1d2d681%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Heinrich-Schacht-Stra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_778537ae6c6349d8a60ed0f349a1255d.setContent%28html_b10f7b0ae9864ce086dd3b7cf1d2d681%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_97295e4073174a2fb1c4e9f06bf9ef7d.bindPopup%28popup_778537ae6c6349d8a60ed0f349a1255d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_484e8662f2a347aba8d945909c1f6c25%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5784133245173%2C9.7196630962585%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2d764380476b418ba8d2364a71650b78%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bd36a0e9a9de4afbadc42e82174ade6d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_bd36a0e9a9de4afbadc42e82174ade6d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Lindenstra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2d764380476b418ba8d2364a71650b78.setContent%28html_bd36a0e9a9de4afbadc42e82174ade6d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_484e8662f2a347aba8d945909c1f6c25.bindPopup%28popup_2d764380476b418ba8d2364a71650b78%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cea955ad6bbb44f5be785e013d04934e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5889794348674%2C9.7067803144455%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_1ab29139cfc84512ab690be5e4eda8f6%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_458f7eb969c64c5c9fdabb569dbe6d72%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_458f7eb969c64c5c9fdabb569dbe6d72%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pferdekoppel%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_1ab29139cfc84512ab690be5e4eda8f6.setContent%28html_458f7eb969c64c5c9fdabb569dbe6d72%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_cea955ad6bbb44f5be785e013d04934e.bindPopup%28popup_1ab29139cfc84512ab690be5e4eda8f6%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ae6c302910a142e79ce9cc381786cfdc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5855916527196%2C9.70323175191879%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_aeaa5f270c9c489eacd73b76b60fdabb%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b3f8f303fbb54bd387ee8f0694e07f96%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b3f8f303fbb54bd387ee8f0694e07f96%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pinneberger%20Stra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_aeaa5f270c9c489eacd73b76b60fdabb.setContent%28html_b3f8f303fbb54bd387ee8f0694e07f96%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ae6c302910a142e79ce9cc381786cfdc.bindPopup%28popup_aeaa5f270c9c489eacd73b76b60fdabb%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_698f533ebe034c0d8e7d3b56a167545c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5810532740654%2C9.70545530319214%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_a729416089fb435d964d03b4170d35ab%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5f567496bd684acebed9e07ab70b5b3d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5f567496bd684acebed9e07ab70b5b3d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rosengarten%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_a729416089fb435d964d03b4170d35ab.setContent%28html_5f567496bd684acebed9e07ab70b5b3d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_698f533ebe034c0d8e7d3b56a167545c.bindPopup%28popup_a729416089fb435d964d03b4170d35ab%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f580a1f9e1014325b46862b0c4a839d5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5989169842047%2C9.73109203352019%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_07182b31f4e146c8af145d72d6b8cd48%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_602beba53357419c8cc153180e375882%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_602beba53357419c8cc153180e375882%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Schwartenseekamp%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_07182b31f4e146c8af145d72d6b8cd48.setContent%28html_602beba53357419c8cc153180e375882%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_f580a1f9e1014325b46862b0c4a839d5.bindPopup%28popup_07182b31f4e146c8af145d72d6b8cd48%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_13e22b4134ac4f96b3c8ed1a5be7be00%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5709480505284%2C9.69663619995117%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_ae78931909e4429c9056f056a02f716e%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fba6274e9653427e9083142c78906435%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_fba6274e9653427e9083142c78906435%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Strandbad%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_ae78931909e4429c9056f056a02f716e.setContent%28html_fba6274e9653427e9083142c78906435%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_13e22b4134ac4f96b3c8ed1a5be7be00.bindPopup%28popup_ae78931909e4429c9056f056a02f716e%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cff8fc43da1d4b88ae03e52d641d1f5b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5751228077681%2C9.70517635345459%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_743524872c7b4150b3663c668e33bf67%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c952d9f562a440b9861e7a0d139e5a28%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c952d9f562a440b9861e7a0d139e5a28%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Vogt-K%C3%B6rner%20Stra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_743524872c7b4150b3663c668e33bf67.setContent%28html_c952d9f562a440b9861e7a0d139e5a28%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_cff8fc43da1d4b88ae03e52d641d1f5b.bindPopup%28popup_743524872c7b4150b3663c668e33bf67%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fe05d8422bd44c6a8b11b4b2204568a7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5907752727735%2C9.69387352466583%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_00ce639cf6924521af146d4a0da916f5%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_11790681440347169f95103b773cfbfe%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_11790681440347169f95103b773cfbfe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wacholderstra%C3%9Fe%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_00ce639cf6924521af146d4a0da916f5.setContent%28html_11790681440347169f95103b773cfbfe%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_fe05d8422bd44c6a8b11b4b2204568a7.bindPopup%28popup_00ce639cf6924521af146d4a0da916f5%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_dbf4c521da514c5689107ffbf06c7da0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5807953065342%2C9.72207427024841%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_45239a250b0841029625f33a7741e75a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cbfec5d0218541449e9da56cdecd38e4%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_cbfec5d0218541449e9da56cdecd38e4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Kronskamp%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_45239a250b0841029625f33a7741e75a.setContent%28html_cbfec5d0218541449e9da56cdecd38e4%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_dbf4c521da514c5689107ffbf06c7da0.bindPopup%28popup_45239a250b0841029625f33a7741e75a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_02109db697ff42ee896c6689abbd6b7c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5888347076387%2C9.69773345623709%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_4d412181f36644a4891839fa424e451e%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9af2c486d15d415b851446a3ba9774e5%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9af2c486d15d415b851446a3ba9774e5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Appelboomtwiete%20Ecke%20Steinberg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_4d412181f36644a4891839fa424e451e.setContent%28html_9af2c486d15d415b851446a3ba9774e5%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_02109db697ff42ee896c6689abbd6b7c.bindPopup%28popup_4d412181f36644a4891839fa424e451e%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_19b4036094574452af562933ec217ea6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5717208544489%2C9.72274482250214%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6641426cfab64a9ba4ca6b850cc00c32%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1d14daee48f74dbc8e4e8970179f1498%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1d14daee48f74dbc8e4e8970179f1498%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Albert-Schweizer%20Schule%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6641426cfab64a9ba4ca6b850cc00c32.setContent%28html_1d14daee48f74dbc8e4e8970179f1498%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_19b4036094574452af562933ec217ea6.bindPopup%28popup_6641426cfab64a9ba4ca6b850cc00c32%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f7222671141d4781bf901010a6c7f124%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5848168731614%2C9.69250559806824%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_49fb3815ddbf4109bd21b5717e4dea38%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6845afda52224e25967747b5ab7ef140%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_6845afda52224e25967747b5ab7ef140%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20B%C3%BCrgerpark%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_49fb3815ddbf4109bd21b5717e4dea38.setContent%28html_6845afda52224e25967747b5ab7ef140%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_f7222671141d4781bf901010a6c7f124.bindPopup%28popup_49fb3815ddbf4109bd21b5717e4dea38%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e14e4f65b7ec4c84804201607b4ac538%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5911799625823%2C9.72104430198669%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_3a201eeb4bc048bc9594f4bd9b74cda7%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9976fb09b2d24b41a3a7eb8701e4a410%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9976fb09b2d24b41a3a7eb8701e4a410%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Egenb%C3%BCttelweg%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_3a201eeb4bc048bc9594f4bd9b74cda7.setContent%28html_9976fb09b2d24b41a3a7eb8701e4a410%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_e14e4f65b7ec4c84804201607b4ac538.bindPopup%28popup_3a201eeb4bc048bc9594f4bd9b74cda7%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_dc1b95396e664165afc0a9899640a8e0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5699401349262%2C9.7113025188446%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_157ec495da2b4f68b3b5831e1a9bf682%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_316f22d3376541fba9c40641604d49f3%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_316f22d3376541fba9c40641604d49f3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Elbstra%C3%9Fe%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_157ec495da2b4f68b3b5831e1a9bf682.setContent%28html_316f22d3376541fba9c40641604d49f3%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_dc1b95396e664165afc0a9899640a8e0.bindPopup%28popup_157ec495da2b4f68b3b5831e1a9bf682%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cf8d3e6f0ac649c396aefba2154445bd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.591999437962%2C9.72143810255561%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_8433818825834938948e60c0a24b1d46%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_566fc30cf7124c6daa3d051715a147fe%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_566fc30cf7124c6daa3d051715a147fe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Gerhart-Hauptmann%20Stra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_8433818825834938948e60c0a24b1d46.setContent%28html_566fc30cf7124c6daa3d051715a147fe%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_cf8d3e6f0ac649c396aefba2154445bd.bindPopup%28popup_8433818825834938948e60c0a24b1d46%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a7934b1e72c54043be945ded82e1c050%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5899384230329%2C9.69445219130904%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_cc06969c60df457397cc3ae4ce73b366%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3b369ac35df8427aa8c2ca9bebdcca7c%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_3b369ac35df8427aa8c2ca9bebdcca7c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hainbuchenweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_cc06969c60df457397cc3ae4ce73b366.setContent%28html_3b369ac35df8427aa8c2ca9bebdcca7c%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a7934b1e72c54043be945ded82e1c050.bindPopup%28popup_cc06969c60df457397cc3ae4ce73b366%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_86a0eae652a34c95925c88f499e559fe%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5669774121414%2C9.72199380397797%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_9245035dc78a4d47bec8bb98ef1fafc4%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_be0155def2f444148b72ef9a342afd76%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_be0155def2f444148b72ef9a342afd76%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hellgrund%20Wedel%20Cluster%202%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_9245035dc78a4d47bec8bb98ef1fafc4.setContent%28html_be0155def2f444148b72ef9a342afd76%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_86a0eae652a34c95925c88f499e559fe.bindPopup%28popup_9245035dc78a4d47bec8bb98ef1fafc4%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1e005baa41a84d64a31670bbb8460573%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5919426661751%2C9.71341580374904%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_fe73554fb513474fbf628a9ed1cb5ff1%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e0ba2843fba3485d9dc203647ad7be65%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_e0ba2843fba3485d9dc203647ad7be65%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Klintkamp%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_fe73554fb513474fbf628a9ed1cb5ff1.setContent%28html_e0ba2843fba3485d9dc203647ad7be65%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_1e005baa41a84d64a31670bbb8460573.bindPopup%28popup_fe73554fb513474fbf628a9ed1cb5ff1%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b7205c5a554f4a8d913e3bd33a48ca5a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5863301162117%2C9.69326734542847%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_dc069c4ece0a4ac293e571c67545e4a1%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5245b9313f7f4ab38a355c25977f92f6%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5245b9313f7f4ab38a355c25977f92f6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Reepschl%C3%A4gerstra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_dc069c4ece0a4ac293e571c67545e4a1.setContent%28html_5245b9313f7f4ab38a355c25977f92f6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b7205c5a554f4a8d913e3bd33a48ca5a.bindPopup%28popup_dc069c4ece0a4ac293e571c67545e4a1%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f3240041a5574b0d9a1af8cf58cab985%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5898009446568%2C9.69020962715149%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d9dfb27ef138449e974082eb0af4a584%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_05509c369432492698dde7b556da6462%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_05509c369432492698dde7b556da6462%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Schlehdornweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d9dfb27ef138449e974082eb0af4a584.setContent%28html_05509c369432492698dde7b556da6462%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_f3240041a5574b0d9a1af8cf58cab985.bindPopup%28popup_d9dfb27ef138449e974082eb0af4a584%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2a75b452fac44c258d82aae04575f393%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.59188443917404%2C9.724164381623268%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6f0736320cd942f09a4aca0688781fea%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_28e74a90d9114d4da1de8a340468a082%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_28e74a90d9114d4da1de8a340468a082%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Von-Suttner%20Stra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6f0736320cd942f09a4aca0688781fea.setContent%28html_28e74a90d9114d4da1de8a340468a082%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_2a75b452fac44c258d82aae04575f393.bindPopup%28popup_6f0736320cd942f09a4aca0688781fea%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_72f026974e4348899e370b980f2a17c4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5884908446434%2C9.70395349985231%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_b38f81007f38401991012f334bafdab6%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1a6ffe6944664eeb8e08a1e55a205dce%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1a6ffe6944664eeb8e08a1e55a205dce%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wiedetwiete%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_b38f81007f38401991012f334bafdab6.setContent%28html_1a6ffe6944664eeb8e08a1e55a205dce%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_72f026974e4348899e370b980f2a17c4.bindPopup%28popup_b38f81007f38401991012f334bafdab6%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ebff2a2c7ff24791beea1b3fb06053fe%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5824138041649%2C9.71122829167579%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_10b9d645e0b240aebbdbfa787552372b%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_de438bd02c534618bf830357a9b7c294%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_de438bd02c534618bf830357a9b7c294%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Autal%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_10b9d645e0b240aebbdbfa787552372b.setContent%28html_de438bd02c534618bf830357a9b7c294%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ebff2a2c7ff24791beea1b3fb06053fe.bindPopup%28popup_10b9d645e0b240aebbdbfa787552372b%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_37a2d43d625a4db59a400d4578fed208%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.575454069497%2C9.7262316942215%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f070fcc8d1934c62928fa644e074ebc7%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9a8f53f96ead44e4aff60d1b9f5d3971%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9a8f53f96ead44e4aff60d1b9f5d3971%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Im%20Grund%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f070fcc8d1934c62928fa644e074ebc7.setContent%28html_9a8f53f96ead44e4aff60d1b9f5d3971%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_37a2d43d625a4db59a400d4578fed208.bindPopup%28popup_f070fcc8d1934c62928fa644e074ebc7%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5e35b0fc0e9a481098366e6273096136%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5695355602878%2C9.70255315303802%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0ca2597d96de45dfa9916b319ea18263%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_936ca7c0f46e43a690cbfa030a6d4d93%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_936ca7c0f46e43a690cbfa030a6d4d93%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Parna%C3%9Fstra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0ca2597d96de45dfa9916b319ea18263.setContent%28html_936ca7c0f46e43a690cbfa030a6d4d93%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_5e35b0fc0e9a481098366e6273096136.bindPopup%28popup_0ca2597d96de45dfa9916b319ea18263%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1fb4669c83a945d8b261d4c5af2317df%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5933545865536%2C9.71821457147598%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_8cbc4f16d9584335b72db951489f4245%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_05b83e1f97cb433b8b3e14e8ee19b022%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_05b83e1f97cb433b8b3e14e8ee19b022%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rebhuhnweg%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_8cbc4f16d9584335b72db951489f4245.setContent%28html_05b83e1f97cb433b8b3e14e8ee19b022%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_1fb4669c83a945d8b261d4c5af2317df.bindPopup%28popup_8cbc4f16d9584335b72db951489f4245%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3285df2f65c74a97a961886f4bb975eb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.59039491694423%2C9.69691358503951%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_107aa96f4efb49c4a1e454c4c6444bf1%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_34be1cb457ce4d9fbc0e2bf6f51b5ece%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_34be1cb457ce4d9fbc0e2bf6f51b5ece%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Appelboomtwiete%20Ecke%20Aastwiete%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_107aa96f4efb49c4a1e454c4c6444bf1.setContent%28html_34be1cb457ce4d9fbc0e2bf6f51b5ece%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_3285df2f65c74a97a961886f4bb975eb.bindPopup%28popup_107aa96f4efb49c4a1e454c4c6444bf1%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5c287817085345b7bfb119e8ef76cf51%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5754046192883%2C9.71914261579514%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_5560cc7c953d4fa899cfea9c88a05cd0%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_795ad8c6b2a546e69df2bf7d9bacce6e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_795ad8c6b2a546e69df2bf7d9bacce6e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Tinsdaler%20Weg%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_5560cc7c953d4fa899cfea9c88a05cd0.setContent%28html_795ad8c6b2a546e69df2bf7d9bacce6e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_5c287817085345b7bfb119e8ef76cf51.bindPopup%28popup_5560cc7c953d4fa899cfea9c88a05cd0%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_90ee34bb8c4741099b8b9eee9a623919%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5822166562335%2C9.70818042755127%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d861705fc37b49ffbf5a4509fddd79c4%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_19ce4cd7679e47ea99b1c412d9346c06%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_19ce4cd7679e47ea99b1c412d9346c06%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Theaterstra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d861705fc37b49ffbf5a4509fddd79c4.setContent%28html_19ce4cd7679e47ea99b1c412d9346c06%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_90ee34bb8c4741099b8b9eee9a623919.bindPopup%28popup_d861705fc37b49ffbf5a4509fddd79c4%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9a55e2ad4a7a43d6a4b98ea3491270b1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5937489838527%2C9.73056882619858%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_895fd36e9fe249838941d19e05f325a1%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_921e2add7bf34de78e18fdf92ee1f5bb%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_921e2add7bf34de78e18fdf92ee1f5bb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Heinestra%C3%9Fe%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_895fd36e9fe249838941d19e05f325a1.setContent%28html_921e2add7bf34de78e18fdf92ee1f5bb%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_9a55e2ad4a7a43d6a4b98ea3491270b1.bindPopup%28popup_895fd36e9fe249838941d19e05f325a1%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d3be0c4c3f1247d29e3751f94d6dd795%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5873395516784%2C9.70869541168213%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23000000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23000000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_86885ca849f34cd7bfe1c34b28696a3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_923f0f75bad84025a8f0ba6095de7bf3%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_56f6ff15a9c841939a752618af98bf00%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_56f6ff15a9c841939a752618af98bf00%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Opn%20Klint%20Wedel%20%2ANote%20not%20clustered%20-%20no%20venues%20listed%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_923f0f75bad84025a8f0ba6095de7bf3.setContent%28html_56f6ff15a9c841939a752618af98bf00%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_d3be0c4c3f1247d29e3751f94d6dd795.bindPopup%28popup_923f0f75bad84025a8f0ba6095de7bf3%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



#### Make detailed lists of each cluster for analysis:


```python
'''
This cluster is around a more residential neighborhood of the village. 
There are less cafes here and a more residential/homeowner services.
'''
final_df.loc[final_df['Cluster Labels'] == 0, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Haselweg Wedel</td>
      <td>53.591291</td>
      <td>Schöner Spielplatz mit angegliederter kleiner...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Supermarket</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>53.591441</td>
      <td>Der Spielplatz macht einen herausragenden Eind...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Anne-Frank-Weg Wedel</td>
      <td>53.588950</td>
      <td>Spielplatz mit Matschanlage (also die Ersatzk...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Garden Center</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Ansgariusweg Wedel</td>
      <td>53.587196</td>
      <td>An der Zufahrt zum Fährmannssand liegt dieser...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Bakery</td>
      <td>Tea Room</td>
      <td>Supermarket</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Ernst-Thälmann-Weg Wedel</td>
      <td>53.588167</td>
      <td>Eingeschlossen von Häusern liegt hier ein ruh...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Garden Center</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pferdekoppel Wedel</td>
      <td>53.588979</td>
      <td>Dieser Spielplatz mit schöner Spielburg liegt...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Supermarket</td>
      <td>College Gym</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Wacholderstraße Wedel</td>
      <td>53.590775</td>
      <td>Spielplatz mit Schwerpunkt Sandspiele.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Tea Room</td>
      <td>Garden Center</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Appelboomtwiete Ecke Steinberg Wedel</td>
      <td>53.588835</td>
      <td>Diesen Spielplatz haben wir noch nicht besuch...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Tea Room</td>
      <td>College Gym</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hainbuchenweg Wedel</td>
      <td>53.589938</td>
      <td>Spielplatz eher für etwas ältere Kinder.&lt;br/&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Tea Room</td>
      <td>Garden Center</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Schlehdornweg Wedel</td>
      <td>53.589801</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Bakery</td>
      <td>Tea Room</td>
      <td>Supermarket</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Wiedetwiete Wedel</td>
      <td>53.588491</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>College Gym</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Appelboomtwiete Ecke Aastwiete Wedel</td>
      <td>53.590395</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Tea Room</td>
      <td>College Gym</td>
      <td>Insurance Office</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''This cluster includes the bulk of observations. 
These playgrounds are generally in the more urban area of the village.
From these locations, users have access to several small shops and services.
They're perhaps a better choice for an extended outing that includes playgrounds and socializing.
'''
final_df.loc[final_df['Cluster Labels'] == 1, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>53.592631</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>Bakery</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>Großer Spielplatz mit viel Wiese. Die Spielge...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Mexican Restaurant</td>
      <td>Supermarket</td>
      <td>Café</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Mühlenweg Wedel</td>
      <td>53.581066</td>
      <td>Schön gestalteter Spielplatz am Mühlenweg.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Bakery</td>
      <td>Drugstore</td>
      <td>Italian Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Gym</td>
      <td>Shopping Mall</td>
      <td>Fast Food Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hamburger Yachthafen Wedel</td>
      <td>53.574397</td>
      <td>neuer, riesiger toller spielplatz, muss man h...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Boat or Ferry</td>
      <td>Harbor / Marina</td>
      <td>Seafood Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Alter Zirkusplatz Wedel</td>
      <td>53.575596</td>
      <td>Versteckter Spielplatz mit schattigen Ecken u...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Supermarket</td>
      <td>Shopping Mall</td>
      <td>Turkish Restaurant</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Bank</td>
      <td>Taverna</td>
      <td>Optical Shop</td>
      <td>Clothing Store</td>
      <td>Bus Stop</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Altstadtschule Wedel</td>
      <td>53.582625</td>
      <td>Dieser Spielplatz liegt auf dem Schulhof der ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Italian Restaurant</td>
      <td>Hotel</td>
      <td>Sculpture Garden</td>
      <td>Trattoria/Osteria</td>
      <td>Museum</td>
      <td>Fast Food Restaurant</td>
      <td>Drugstore</td>
      <td>Doner Restaurant</td>
      <td>Pool</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Brombeerweg Wedel</td>
      <td>53.574119</td>
      <td>Schöner kleiner Spielplatz unter Bäumen&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Bus Stop</td>
      <td>Café</td>
      <td>Food &amp; Drink Shop</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Croningstraße Wedel</td>
      <td>53.581910</td>
      <td>Dieser Spielplatz nur von der Croningstraße e...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Supermarket</td>
      <td>Fast Food Restaurant</td>
      <td>Furniture / Home Store</td>
      <td>Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Pet Store</td>
      <td>French Restaurant</td>
      <td>Nightclub</td>
      <td>Sandwich Place</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Gärtnerstraße Wedel</td>
      <td>53.585607</td>
      <td>Dieser Spielplatz wurde seit meinem letzten B...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Hotel</td>
      <td>Italian Restaurant</td>
      <td>Sculpture Garden</td>
      <td>Trattoria/Osteria</td>
      <td>Museum</td>
      <td>College Gym</td>
      <td>Pub</td>
      <td>German Restaurant</td>
      <td>Café</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Heinrich-Schacht-Straße Wedel</td>
      <td>53.580021</td>
      <td>Ein größerer Spielplatz. Bemerkenswert neben ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>Supermarket</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>French Restaurant</td>
      <td>Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Taverna</td>
      <td>Bakery</td>
      <td>Beach</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Lindenstraße Wedel</td>
      <td>53.578413</td>
      <td>Ein langezogener Spielplatz mit insgesamt vie...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Taverna</td>
      <td>Bus Stop</td>
      <td>French Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pinneberger Straße Wedel</td>
      <td>53.585592</td>
      <td>Abgegrenzt vom Obstbaumweg an der Rückseite u...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Italian Restaurant</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>German Restaurant</td>
      <td>Trattoria/Osteria</td>
      <td>Doner Restaurant</td>
      <td>Pub</td>
      <td>College Gym</td>
      <td>Sculpture Garden</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Rosengarten Wedel</td>
      <td>53.581053</td>
      <td>Dieser Spielplatz liegt am Rosengarten, nahe ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Drugstore</td>
      <td>Turkish Restaurant</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Doner Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>German Restaurant</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Schwartenseekamp Wedel</td>
      <td>53.598917</td>
      <td>Diesen Spielplatz haben wir noch nicht besuch...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Spa</td>
      <td>Plaza</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Strandbad Wedel</td>
      <td>53.570948</td>
      <td>Großer Spielplatz:&lt;br/&gt;Wegen des Wassers in d...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Seafood Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Beach</td>
      <td>Beach Bar</td>
      <td>Soccer Field</td>
      <td>Pier</td>
      <td>Hotel</td>
      <td>Fast Food Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Vogt-Körner Straße Wedel</td>
      <td>53.575123</td>
      <td>Dieser kleine Spielplatz liegt versteckt hint...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Supermarket</td>
      <td>Turkish Restaurant</td>
      <td>Optical Shop</td>
      <td>Café</td>
      <td>Shopping Mall</td>
      <td>Clothing Store</td>
      <td>Drugstore</td>
      <td>Fast Food Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Kronskamp Wedel</td>
      <td>53.580795</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Supermarket</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>French Restaurant</td>
      <td>Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Taverna</td>
      <td>Bakery</td>
      <td>Beach</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Bürgerpark Wedel</td>
      <td>53.584817</td>
      <td>Kleiner Spielplatz im Bürgerpark.</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Italian Restaurant</td>
      <td>Supermarket</td>
      <td>Sculpture Garden</td>
      <td>Museum</td>
      <td>Steakhouse</td>
      <td>Hotel</td>
      <td>Theater</td>
      <td>Pub</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Egenbüttelweg Wedel</td>
      <td>53.591180</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Turkish Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Gerhart-Hauptmann Straße Wedel</td>
      <td>53.591999</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Turkish Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Klintkamp Wedel</td>
      <td>53.591943</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Mexican Restaurant</td>
      <td>Supermarket</td>
      <td>Café</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Reepschlägerstraße Wedel</td>
      <td>53.586330</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Italian Restaurant</td>
      <td>Supermarket</td>
      <td>Garden Center</td>
      <td>Museum</td>
      <td>Pub</td>
      <td>Sculpture Garden</td>
      <td>Steakhouse</td>
      <td>Hotel</td>
      <td>Beach Bar</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Von-Suttner Straße Wedel</td>
      <td>53.591884</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Bakery</td>
      <td>Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Autal Wedel</td>
      <td>53.582414</td>
      <td>Sehr kleiner Spielplatz. Gelegen nahe an der ...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Thai Restaurant</td>
      <td>Gym</td>
      <td>Sushi Restaurant</td>
      <td>Steakhouse</td>
      <td>Bus Stop</td>
      <td>Doner Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Im Grund Wedel</td>
      <td>53.575454</td>
      <td>Zunächst findet man hier den Bolzplatz, dahin...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>Restaurant</td>
      <td>Bus Stop</td>
      <td>Café</td>
      <td>Food &amp; Drink Shop</td>
      <td>Turkish Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Parnaßstraße Wedel</td>
      <td>53.569536</td>
      <td>Kleiner Spielplatz im Parnaßpark, dem die Gra...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Seafood Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Beach</td>
      <td>Beach Bar</td>
      <td>Bus Stop</td>
      <td>Pier</td>
      <td>Hotel</td>
      <td>Fast Food Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Rebhuhnweg Wedel</td>
      <td>53.593355</td>
      <td>Kleiner Spielplatz&lt;br/&gt;</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Mexican Restaurant</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Tinsdaler Weg Wedel</td>
      <td>53.575405</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Taverna</td>
      <td>Garden</td>
      <td>Photography Studio</td>
      <td>Turkish Restaurant</td>
      <td>Drugstore</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Theaterstraße Wedel</td>
      <td>53.582217</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Drugstore</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Trattoria/Osteria</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Taverna</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Heinestraße Wedel</td>
      <td>53.593749</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Bakery</td>
      <td>Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
This cluster isn't the only beach playground on the list. But, it's in a cluster of its own due to 
isolation from other shops.
'''
final_df.loc[final_df['Cluster Labels'] == 2, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Hellgrund Wedel</td>
      <td>53.566977</td>
      <td>Versteckt im Tal in direkter Nähe zum Vattenf...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Beach</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
It is intereting that this group received it's own cluster rather than being grouped with the large cluster.
It definitely has a different, distinctive feel from the ceneter of the village. The area appears to have been
redeveloped in the 1960's-70's and is somehow being reflected as different in the shops available.
'''
final_df.loc[final_df['Cluster Labels'] == 3, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Ginsterweg Wedel</td>
      <td>53.573638</td>
      <td>Der Spielplatz ist auf mehrere Ebenen in eine...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>Taverna</td>
      <td>Supermarket</td>
      <td>Garden</td>
      <td>Bus Stop</td>
      <td>Photography Studio</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hans-Böckler Platz Wedel</td>
      <td>53.568860</td>
      <td>Der Spielplatz ist zur Straße hin mit einem Z...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Bakery</td>
      <td>Beach</td>
      <td>Supermarket</td>
      <td>Bus Stop</td>
      <td>Turkish Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pulverstraße Wedel</td>
      <td>53.571205</td>
      <td>Dieser mittelgroße Spielplatz befindet sich n...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>Bakery</td>
      <td>Supermarket</td>
      <td>Bus Stop</td>
      <td>Garden</td>
      <td>Turkish Restaurant</td>
      <td>Electronics Store</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Albert-Schweizer Schule Wedel</td>
      <td>53.571721</td>
      <td>Dieser Spielplatz befindet sich auf dem Gelän...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Garden</td>
      <td>Supermarket</td>
      <td>Photography Studio</td>
      <td>Bus Stop</td>
      <td>Electronics Store</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Elbstraße Wedel</td>
      <td>53.569940</td>
      <td>Kleiner Spielplatz.&lt;br/&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Bus Stop</td>
      <td>Bakery</td>
      <td>Beach</td>
      <td>Supermarket</td>
      <td>Turkish Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
This cluster is just a pair of playgrounds on the road into the village. They just share the same shops.
'''
final_df.loc[final_df['Cluster Labels'] == 4, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Rotdornstraße Wedel</td>
      <td>53.591067</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Insurance Office</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Geesthang Wedel</td>
      <td>53.589786</td>
      <td>Dieser Spielplatz befindet sich am Ende der H...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>Garden Center</td>
      <td>Turkish Restaurant</td>
      <td>Insurance Office</td>
      <td>Harbor / Marina</td>
      <td>Gym / Fitness Center</td>
      <td>Gym</td>
      <td>German Restaurant</td>
      <td>Garden</td>
      <td>Furniture / Home Store</td>
      <td>French Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
Finally, if any playgrounds didn't return commercial venues nearby from Foursquare they would end up in this list.
These would perhaps still be great playgrounds, but inconvenient if trying to do a combined play-shopping trip.
Alternatively, Foursquare is not entirely consistent for this area as this playground was grouped in prior runs.
'''
exception_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster Labels</th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_long</th>
      <th>a_name_address</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Spielplatz Opn Klint Wedel</td>
      <td>53.58734</td>
      <td>9.708695</td>
      <td>Dieser Spielplatz ist zum Teil öffentlch, zum ...</td>
      <td>Dieser Spielplatz ist zum Teil öffentlch, zum...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Part IIC: Clustering and mapping based on playground equipment <a name="bonus1"></a>
This is an alternative way to map and explore that data. If the focus isn't on combining tasks, but rather choosing a
playground based on the equipment available, clustering can help reduce the search cost involved.

#### Prepare a second dataframe to use in clustering on playground equipment:


```python
#Start by copying the dataframe that has playground equipment noted but not surrounding venues:
df_alt=p_df.copy()
#Copy the columns that will be dropped when clustering so they can be added back afterwards: 
df_labels=df_alt[['a_name','a_lat','a_long','a_name_address','a_description','a_rating']].copy()
#Drop the columns not needed in clustering:
df_alt=df_alt.drop(['a_lat','a_long','a_name_address','a_description','a_rating'], axis=1)
#Rename and keep the Playground name column for merging later:
df_alt=df_alt.rename(columns={"a_name":"Playground"})
df_row_names=df_alt['Playground'].copy()
df_alt=df_alt.drop(['Playground'], axis=1)
#Normalize the data using StandardScaler:
normalized_DS=StandardScaler().fit_transform(df_alt)
print(normalized_DS)
df_alt.head(10)
```

    [[-0.29172998 -0.14142136  4.94974747 ...  0.          0.
       3.50051626]
     [-0.29172998 -0.14142136 -0.20203051 ...  0.          0.
       1.19198614]
     [-0.29172998 -0.14142136 -0.20203051 ...  0.          0.
       0.80723112]
     ...
     [-0.29172998 -0.14142136 -0.20203051 ...  0.          0.
      -1.501299  ]
     [-0.29172998 -0.14142136 -0.20203051 ...  0.          0.
      -1.501299  ]
     [-0.29172998 -0.14142136 -0.20203051 ...  0.          0.
      -1.501299  ]]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



#### Run k-means clustering on the playground equipment data:


```python
#Set number of clusters
kclusters = 5
#Run k-means clustering:
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(normalized_DS)
#Check cluster labels generated for each row in the dataframe:
kmeans.labels_[0:len(df_alt)]
```




    array([1, 0, 0, 2, 0, 2, 4, 0, 0, 0, 0, 4, 2, 3, 0, 0, 2, 0, 3, 3, 4, 4,
           4, 0, 0, 3, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 3, 4, 3, 4, 4, 0, 4, 4,
           3, 4, 4, 4, 4, 4, 4])



#### Combine clustering result and datasets:


```python
#Add the playground names:
df_alt['Playground']=df_row_names
#Add the cluster labels:
df_alt.insert(0, 'Cluster Labels', kmeans.labels_)
#Prepare the datasets for merging by 'playground':
df_alt=df_alt.rename(columns={"a_name":"Playground"})
df_labels=df_labels.rename(columns={"a_name":"Playground"})
#Merge/join the datasets:
final_df = df_labels.join(df_alt.set_index('Playground'), on='Playground')
#Check the dataframe shape:
print(final_df.shape)
#Move cluster labels to the front:
first_column = final_df.pop('Cluster Labels')
final_df.insert(0, 'Cluster Labels', first_column)
#Ensure cluster labesl are still integers (changes if any were blank):
final_df['Cluster Labels']=final_df['Cluster Labels'].astype(int)
#Display the dataframe:
final_df.head()
```

    (51, 35)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster Labels</th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_long</th>
      <th>a_name_address</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>53.592631</td>
      <td>9.731698</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Haselweg Wedel</td>
      <td>53.591291</td>
      <td>9.706469</td>
      <td>Schöner Spielplatz mit angegliederter kleiner ...</td>
      <td>Schöner Spielplatz mit angegliederter kleiner...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>9.715068</td>
      <td>Großer Spielplatz mit viel Wiese. Die Spielger...</td>
      <td>Großer Spielplatz mit viel Wiese. Die Spielge...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>53.591441</td>
      <td>9.705530</td>
      <td>Spielplatz Wasserspielplatz Haus am See in Wed...</td>
      <td>Der Spielplatz macht einen herausragenden Eind...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Spielplatz Mühlenweg Wedel</td>
      <td>53.581066</td>
      <td>9.710192</td>
      <td>Schön gestalteter Spielplatz am Mühlenweg.</td>
      <td>Schön gestalteter Spielplatz am Mühlenweg.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



#### Map the new clusters:


```python
#This follows the same process discussed in the last two mapping cells above.
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=13)
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
# add markers to the map:
for lat, lon, poi, cluster in zip(final_df['a_lat'], final_df['a_long'], final_df['Playground'], final_df['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters) 
'''
Note I do not expect any particular clustering pattern to emerge in the map. The map is useful, however, in
determining which playgrounds with certain equipment groups are available near a given point.
'''
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%3Cscript%3EL_PREFER_CANVAS%20%3D%20false%3B%20L_NO_TOUCH%20%3D%20false%3B%20L_DISABLE_3D%20%3D%20false%3B%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawgit.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%20%23map_a85d11c71d094812aef2dfc89f659ff6%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%20%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%20%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_a85d11c71d094812aef2dfc89f659ff6%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20bounds%20%3D%20null%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_a85d11c71d094812aef2dfc89f659ff6%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27map_a85d11c71d094812aef2dfc89f659ff6%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7Bcenter%3A%20%5B53.5810226%2C9.7038772%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2013%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20maxBounds%3A%20bounds%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20layers%3A%20%5B%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20worldCopyJump%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_27b12b9bcb8d48619cec6b68068a1f0e%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22attribution%22%3A%20null%2C%0A%20%20%22detectRetina%22%3A%20false%2C%0A%20%20%22maxZoom%22%3A%2018%2C%0A%20%20%22minZoom%22%3A%201%2C%0A%20%20%22noWrap%22%3A%20false%2C%0A%20%20%22subdomains%22%3A%20%22abc%22%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ece3273981024b3499283f3885aceb3c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5926308917772%2C9.73169803619385%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%238000ff%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%238000ff%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f1faab6258564b78afdde360f3c3225e%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9dad8323306745d9a3d79dd36c33fc6a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9dad8323306745d9a3d79dd36c33fc6a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Waldspielplatz%20Moorwegsiedlung%20Wedel%20Cluster%201%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f1faab6258564b78afdde360f3c3225e.setContent%28html_9dad8323306745d9a3d79dd36c33fc6a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ece3273981024b3499283f3885aceb3c.bindPopup%28popup_f1faab6258564b78afdde360f3c3225e%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_24f63013cf84411396201145d8650ccb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5912910844463%2C9.70646917819977%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_15bb8df02c844d4392ab79cd8f16659d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1912df998f0a4f9697342ad03aa81ed7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1912df998f0a4f9697342ad03aa81ed7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Haselweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_15bb8df02c844d4392ab79cd8f16659d.setContent%28html_1912df998f0a4f9697342ad03aa81ed7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_24f63013cf84411396201145d8650ccb.bindPopup%28popup_15bb8df02c844d4392ab79cd8f16659d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b3fa702d35b74e1d84ca693300fc5df7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5943062279335%2C9.71506834030151%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_b6f214da9d8144cfa9c114a05486e46e%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ecc1ea2e75f24d9eabf475bc7b450701%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_ecc1ea2e75f24d9eabf475bc7b450701%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Meisenweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_b6f214da9d8144cfa9c114a05486e46e.setContent%28html_ecc1ea2e75f24d9eabf475bc7b450701%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b3fa702d35b74e1d84ca693300fc5df7.bindPopup%28popup_b6f214da9d8144cfa9c114a05486e46e%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_776a2e837c7b450bb955eb488ca5fcaa%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5914407324793%2C9.70553040504456%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_a33f8759d28d47cb8412798691fb9d80%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5e2c45becc2a4889b6f446b7219cfbf2%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5e2c45becc2a4889b6f446b7219cfbf2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wasserspielplatz%20Haus%20am%20See%20Wedel%20Cluster%202%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_a33f8759d28d47cb8412798691fb9d80.setContent%28html_5e2c45becc2a4889b6f446b7219cfbf2%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_776a2e837c7b450bb955eb488ca5fcaa.bindPopup%28popup_a33f8759d28d47cb8412798691fb9d80%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_05a7ff9b89f24b718109941a1cbdaadd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.581066013162%2C9.71019208431244%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_49a3bd49305f45ecb47582667092916d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d51ce2c58d534fab973d845e8c5940e6%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_d51ce2c58d534fab973d845e8c5940e6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20M%C3%BChlenweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_49a3bd49305f45ecb47582667092916d.setContent%28html_d51ce2c58d534fab973d845e8c5940e6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_05a7ff9b89f24b718109941a1cbdaadd.bindPopup%28popup_49a3bd49305f45ecb47582667092916d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7d1753f507764e1d82b90e02b2f79a56%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.591066930019%2C9.68836158514023%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_548646aa72e94653b72ad8bd76b8e412%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f86dd79924314aed8836dba11fb674e7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_f86dd79924314aed8836dba11fb674e7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rotdornstra%C3%9Fe%20Wedel%20Cluster%202%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_548646aa72e94653b72ad8bd76b8e412.setContent%28html_f86dd79924314aed8836dba11fb674e7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_7d1753f507764e1d82b90e02b2f79a56.bindPopup%28popup_548646aa72e94653b72ad8bd76b8e412%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ca1047cdd3da4f16b503b6a4f6984af1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5743968895724%2C9.68239367008209%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_5cb1c6c041d44e59a784f0967a7058cc%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_75c1dec376664843b0621b4ff757437c%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_75c1dec376664843b0621b4ff757437c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hamburger%20Yachthafen%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_5cb1c6c041d44e59a784f0967a7058cc.setContent%28html_75c1dec376664843b0621b4ff757437c%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ca1047cdd3da4f16b503b6a4f6984af1.bindPopup%28popup_5cb1c6c041d44e59a784f0967a7058cc%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b502d81885ad4d33a7facbb0ba9b2ba3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5736384685368%2C9.71911311149597%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2ada2934202246c791f9a3e90a747655%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1d2639ad2b784356b225227b17df7596%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1d2639ad2b784356b225227b17df7596%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ginsterweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2ada2934202246c791f9a3e90a747655.setContent%28html_1d2639ad2b784356b225227b17df7596%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b502d81885ad4d33a7facbb0ba9b2ba3.bindPopup%28popup_2ada2934202246c791f9a3e90a747655%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d0d7c5b188ba4a0881e7f00d73660a35%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5688601987249%2C9.71491277217865%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_3e64081b4fd24a76967bc8249399512d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a80bec649c644fb289fd4a5e809be77c%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_a80bec649c644fb289fd4a5e809be77c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hans-B%C3%B6ckler%20Platz%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_3e64081b4fd24a76967bc8249399512d.setContent%28html_a80bec649c644fb289fd4a5e809be77c%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_d0d7c5b188ba4a0881e7f00d73660a35.bindPopup%28popup_3e64081b4fd24a76967bc8249399512d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5911ca3e17344d5998d3437280ef2157%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5712051224608%2C9.71940010786057%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_dbbb3be01a324f4894b61d870baf4dbb%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6b5ffeb8dfe24d979bd2ca26ea84631a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_6b5ffeb8dfe24d979bd2ca26ea84631a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pulverstra%C3%9Fe%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_dbbb3be01a324f4894b61d870baf4dbb.setContent%28html_6b5ffeb8dfe24d979bd2ca26ea84631a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_5911ca3e17344d5998d3437280ef2157.bindPopup%28popup_dbbb3be01a324f4894b61d870baf4dbb%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a306b46f8c494b83b6926962319fbd00%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5755961290153%2C9.71072316169739%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_968e738c4c9246fe9697f7520ffdd279%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_908cd81ab5364403a948598e1aaff21f%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_908cd81ab5364403a948598e1aaff21f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Alter%20Zirkusplatz%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_968e738c4c9246fe9697f7520ffdd279.setContent%28html_908cd81ab5364403a948598e1aaff21f%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a306b46f8c494b83b6926962319fbd00.bindPopup%28popup_968e738c4c9246fe9697f7520ffdd279%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_652f209b09ab452a88af257c5cf4894c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.582625249582%2C9.69926744699478%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d17d7c394e9647729960a1ec53ae13ff%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e21641b04f4841e0b407b85b13a9b213%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_e21641b04f4841e0b407b85b13a9b213%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Altstadtschule%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d17d7c394e9647729960a1ec53ae13ff.setContent%28html_e21641b04f4841e0b407b85b13a9b213%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_652f209b09ab452a88af257c5cf4894c.bindPopup%28popup_d17d7c394e9647729960a1ec53ae13ff%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cc3ecb7dbeee40f6893b072535e02572%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5889495035841%2C9.69376623630524%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_30509dd165fb4df8bdaa1126f6160f1d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0dd0f95962ff4b99aef66d9ca787240c%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_0dd0f95962ff4b99aef66d9ca787240c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Anne-Frank-Weg%20Wedel%20Cluster%202%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_30509dd165fb4df8bdaa1126f6160f1d.setContent%28html_0dd0f95962ff4b99aef66d9ca787240c%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_cc3ecb7dbeee40f6893b072535e02572.bindPopup%28popup_30509dd165fb4df8bdaa1126f6160f1d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0470ec0102374e12985bae9a326d51b4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5871962578912%2C9.68622386455536%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_7eab3ecc03d74544975f1b68d7b0d6dd%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eafeb704b0474e909cc9820c52ebe17b%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_eafeb704b0474e909cc9820c52ebe17b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ansgariusweg%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_7eab3ecc03d74544975f1b68d7b0d6dd.setContent%28html_eafeb704b0474e909cc9820c52ebe17b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0470ec0102374e12985bae9a326d51b4.bindPopup%28popup_7eab3ecc03d74544975f1b68d7b0d6dd%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2fb38a94a69f48e9bda264030390c76d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5741194511191%2C9.72591519355774%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_81a52d9b322446e39ab0c47170c913a7%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1bc5ef3be05b450dac88a864211b372a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1bc5ef3be05b450dac88a864211b372a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Brombeerweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_81a52d9b322446e39ab0c47170c913a7.setContent%28html_1bc5ef3be05b450dac88a864211b372a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_2fb38a94a69f48e9bda264030390c76d.bindPopup%28popup_81a52d9b322446e39ab0c47170c913a7%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_26e88a64257e4cc294dc52fb127f4d28%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5819099697564%2C9.72337782382965%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d3c51ed024564dbda795a32aa5ce9342%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_63895d10dbb44b3d89e2c747be7057fd%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_63895d10dbb44b3d89e2c747be7057fd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Croningstra%C3%9Fe%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d3c51ed024564dbda795a32aa5ce9342.setContent%28html_63895d10dbb44b3d89e2c747be7057fd%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_26e88a64257e4cc294dc52fb127f4d28.bindPopup%28popup_d3c51ed024564dbda795a32aa5ce9342%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7ca48558734146e6b5f613ce7d7ab1fd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5856072564417%2C9.69738721847534%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2300b5eb%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_96837db3f7e14570b7c6213f7bf54d60%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e16954b8e0cc4d228f1106cc149d5b4e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_e16954b8e0cc4d228f1106cc149d5b4e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20G%C3%A4rtnerstra%C3%9Fe%20Wedel%20Cluster%202%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_96837db3f7e14570b7c6213f7bf54d60.setContent%28html_e16954b8e0cc4d228f1106cc149d5b4e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_7ca48558734146e6b5f613ce7d7ab1fd.bindPopup%28popup_96837db3f7e14570b7c6213f7bf54d60%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ce787fd1969d424ba4a6219c94ff32ab%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5881674618245%2C9.69368040561676%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_be47f2258d3d47bfa65bc943d15b28eb%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6b864909ba9740ec8f49ac3ecd31ceb3%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_6b864909ba9740ec8f49ac3ecd31ceb3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Ernst-Th%C3%A4lmann-Weg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_be47f2258d3d47bfa65bc943d15b28eb.setContent%28html_6b864909ba9740ec8f49ac3ecd31ceb3%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ce787fd1969d424ba4a6219c94ff32ab.bindPopup%28popup_be47f2258d3d47bfa65bc943d15b28eb%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c8d1a4bcacc24748b1c6bfee3eca6088%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5897862207119%2C9.68335154549268%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d531b2e3645141558d1f91d56f6dabbd%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ee10074a612b4d0dbc13769080d78db5%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_ee10074a612b4d0dbc13769080d78db5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Geesthang%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d531b2e3645141558d1f91d56f6dabbd.setContent%28html_ee10074a612b4d0dbc13769080d78db5%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_c8d1a4bcacc24748b1c6bfee3eca6088.bindPopup%28popup_d531b2e3645141558d1f91d56f6dabbd%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f4a724e7f1f84d50ac93f83558259360%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5800213944949%2C9.72487986087799%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0b8504ddcbee449f9d7cdb48765c065f%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a95fa59c86f943c1b3a4aac87be5e809%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_a95fa59c86f943c1b3a4aac87be5e809%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Heinrich-Schacht-Stra%C3%9Fe%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0b8504ddcbee449f9d7cdb48765c065f.setContent%28html_a95fa59c86f943c1b3a4aac87be5e809%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_f4a724e7f1f84d50ac93f83558259360.bindPopup%28popup_0b8504ddcbee449f9d7cdb48765c065f%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2dabf35076ad4101b35587c3467985be%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5784133245173%2C9.7196630962585%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_811fd5aa039b4cc4a8803447c561bc16%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8d180fb262e743919d81faee26299c7d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_8d180fb262e743919d81faee26299c7d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Lindenstra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_811fd5aa039b4cc4a8803447c561bc16.setContent%28html_8d180fb262e743919d81faee26299c7d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_2dabf35076ad4101b35587c3467985be.bindPopup%28popup_811fd5aa039b4cc4a8803447c561bc16%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_950857d061074ffb97b5b8d1f9358f21%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5889794348674%2C9.7067803144455%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_355fe4b513f24e8aba063ed70a9afcc9%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5836a8d852f2497dbfa717078bf452be%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5836a8d852f2497dbfa717078bf452be%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pferdekoppel%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_355fe4b513f24e8aba063ed70a9afcc9.setContent%28html_5836a8d852f2497dbfa717078bf452be%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_950857d061074ffb97b5b8d1f9358f21.bindPopup%28popup_355fe4b513f24e8aba063ed70a9afcc9%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ac4cc989da3a4f5999cf1dd3ab1bff8d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5855916527196%2C9.70323175191879%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0010b397680147b7920f36ef770df350%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d4c5b77b5eb14e518de9915bda2567aa%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_d4c5b77b5eb14e518de9915bda2567aa%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Pinneberger%20Stra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0010b397680147b7920f36ef770df350.setContent%28html_d4c5b77b5eb14e518de9915bda2567aa%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ac4cc989da3a4f5999cf1dd3ab1bff8d.bindPopup%28popup_0010b397680147b7920f36ef770df350%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_44302e32fa2f41719df403f67be8a89f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5810532740654%2C9.70545530319214%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_02389f4ee5d244908c5e34deabfe5257%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0736834dbca9482fbc913030f39cb1b2%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_0736834dbca9482fbc913030f39cb1b2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rosengarten%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_02389f4ee5d244908c5e34deabfe5257.setContent%28html_0736834dbca9482fbc913030f39cb1b2%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_44302e32fa2f41719df403f67be8a89f.bindPopup%28popup_02389f4ee5d244908c5e34deabfe5257%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_59a33a0947c84444a14918c09578d467%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5989169842047%2C9.73109203352019%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c32115d6d6244312ad064a62751302f3%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4832c2cb5b6a4a3184f69a5e623e71d2%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_4832c2cb5b6a4a3184f69a5e623e71d2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Schwartenseekamp%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c32115d6d6244312ad064a62751302f3.setContent%28html_4832c2cb5b6a4a3184f69a5e623e71d2%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_59a33a0947c84444a14918c09578d467.bindPopup%28popup_c32115d6d6244312ad064a62751302f3%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0ced31b179d543a79157628f1cdf1bd3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5709480505284%2C9.69663619995117%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c1d536ff6c14444c8504721eebd3b20b%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_129725568b824fd2abda1d751c9c7885%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_129725568b824fd2abda1d751c9c7885%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Strandbad%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c1d536ff6c14444c8504721eebd3b20b.setContent%28html_129725568b824fd2abda1d751c9c7885%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0ced31b179d543a79157628f1cdf1bd3.bindPopup%28popup_c1d536ff6c14444c8504721eebd3b20b%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6f6476cedd824731993d0194a6dfaa80%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5751228077681%2C9.70517635345459%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_32c9fcdcfb4b4799a8e76305d6d34615%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_53b5270c55b14cd791d3b36bc6529582%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_53b5270c55b14cd791d3b36bc6529582%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Vogt-K%C3%B6rner%20Stra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_32c9fcdcfb4b4799a8e76305d6d34615.setContent%28html_53b5270c55b14cd791d3b36bc6529582%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_6f6476cedd824731993d0194a6dfaa80.bindPopup%28popup_32c9fcdcfb4b4799a8e76305d6d34615%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e7a4774be43b4d12b4459101a54a414b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5907752727735%2C9.69387352466583%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6e5b1d25fa6b482c904d5e62fcf52f68%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0101bfbc87ea414d89d9e2b7e3eb3096%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_0101bfbc87ea414d89d9e2b7e3eb3096%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wacholderstra%C3%9Fe%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6e5b1d25fa6b482c904d5e62fcf52f68.setContent%28html_0101bfbc87ea414d89d9e2b7e3eb3096%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_e7a4774be43b4d12b4459101a54a414b.bindPopup%28popup_6e5b1d25fa6b482c904d5e62fcf52f68%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bc917d0e25d248f6973076527dda1070%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5807953065342%2C9.72207427024841%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_067dbbfd88f9418bae9455bc6eeae3c8%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ce5c687020c54a2d913b5f168153ae4c%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_ce5c687020c54a2d913b5f168153ae4c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Kronskamp%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_067dbbfd88f9418bae9455bc6eeae3c8.setContent%28html_ce5c687020c54a2d913b5f168153ae4c%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_bc917d0e25d248f6973076527dda1070.bindPopup%28popup_067dbbfd88f9418bae9455bc6eeae3c8%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_77212d05af1e4a8fae677aa868474ce8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5888347076387%2C9.69773345623709%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2e904be877a94bb6ba475e3ea182b65f%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9071bf4673404c378a57e1100b3e7d76%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9071bf4673404c378a57e1100b3e7d76%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Appelboomtwiete%20Ecke%20Steinberg%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2e904be877a94bb6ba475e3ea182b65f.setContent%28html_9071bf4673404c378a57e1100b3e7d76%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_77212d05af1e4a8fae677aa868474ce8.bindPopup%28popup_2e904be877a94bb6ba475e3ea182b65f%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9bdf3fde8a1b4c0eb8b9f0bbefb31093%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5717208544489%2C9.72274482250214%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6b0f975b028e4fae98cc6cc008ebc869%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_412e7a01e03340c58870eac076ecb14a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_412e7a01e03340c58870eac076ecb14a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Albert-Schweizer%20Schule%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6b0f975b028e4fae98cc6cc008ebc869.setContent%28html_412e7a01e03340c58870eac076ecb14a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_9bdf3fde8a1b4c0eb8b9f0bbefb31093.bindPopup%28popup_6b0f975b028e4fae98cc6cc008ebc869%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9e6df72c1b2340ccad72918e825e7a60%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5848168731614%2C9.69250559806824%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_1d1111732f4b41959566e98f7ff8fae9%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c4be8c38479e4dbbae6ae5de60511e01%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c4be8c38479e4dbbae6ae5de60511e01%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20B%C3%BCrgerpark%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_1d1111732f4b41959566e98f7ff8fae9.setContent%28html_c4be8c38479e4dbbae6ae5de60511e01%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_9e6df72c1b2340ccad72918e825e7a60.bindPopup%28popup_1d1111732f4b41959566e98f7ff8fae9%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4bd704bc823340a7ae2c8bf54f3f9e4e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5911799625823%2C9.72104430198669%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_15c600412a52473a85250f19b6aa8ff8%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c72a198f40434201b6694b680c80953b%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c72a198f40434201b6694b680c80953b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Egenb%C3%BCttelweg%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_15c600412a52473a85250f19b6aa8ff8.setContent%28html_c72a198f40434201b6694b680c80953b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_4bd704bc823340a7ae2c8bf54f3f9e4e.bindPopup%28popup_15c600412a52473a85250f19b6aa8ff8%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_99aca7bc85c14c44ae1bbd52e1de4c3b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5699401349262%2C9.7113025188446%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_77e931f0551040a7a1e78c10eb740338%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b0434bf951d542c6a3f368505d9396cb%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b0434bf951d542c6a3f368505d9396cb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Elbstra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_77e931f0551040a7a1e78c10eb740338.setContent%28html_b0434bf951d542c6a3f368505d9396cb%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_99aca7bc85c14c44ae1bbd52e1de4c3b.bindPopup%28popup_77e931f0551040a7a1e78c10eb740338%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3475427a790b494ea2aa72bd3b589055%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.591999437962%2C9.72143810255561%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_85eb67fe9cc844e6b430bd6ec2040d7d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c2f475377b4741159c0cf0c810e0574e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c2f475377b4741159c0cf0c810e0574e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Gerhart-Hauptmann%20Stra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_85eb67fe9cc844e6b430bd6ec2040d7d.setContent%28html_c2f475377b4741159c0cf0c810e0574e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_3475427a790b494ea2aa72bd3b589055.bindPopup%28popup_85eb67fe9cc844e6b430bd6ec2040d7d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_583df0c476934e5f8548c1790ffd71fe%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5899384230329%2C9.69445219130904%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_8094d0844bbc485f98ff6bea80e12d16%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_603e3391476b44aea3cd111a77881ecc%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_603e3391476b44aea3cd111a77881ecc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hainbuchenweg%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_8094d0844bbc485f98ff6bea80e12d16.setContent%28html_603e3391476b44aea3cd111a77881ecc%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_583df0c476934e5f8548c1790ffd71fe.bindPopup%28popup_8094d0844bbc485f98ff6bea80e12d16%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_76498f63d9ce4667aefc6dd119f53e63%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5669774121414%2C9.72199380397797%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_1630225c712b4572b81bb5c4a3c580ee%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7818a9a5ffb44b3f99da25d384bd237a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_7818a9a5ffb44b3f99da25d384bd237a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Hellgrund%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_1630225c712b4572b81bb5c4a3c580ee.setContent%28html_7818a9a5ffb44b3f99da25d384bd237a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_76498f63d9ce4667aefc6dd119f53e63.bindPopup%28popup_1630225c712b4572b81bb5c4a3c580ee%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_be7db799263445d4a7d8130967231176%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5919426661751%2C9.71341580374904%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_41e73991a9f34578886b659a08a29b98%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_49f9e6c92087483da4d1fd5d2a99b48e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_49f9e6c92087483da4d1fd5d2a99b48e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Klintkamp%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_41e73991a9f34578886b659a08a29b98.setContent%28html_49f9e6c92087483da4d1fd5d2a99b48e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_be7db799263445d4a7d8130967231176.bindPopup%28popup_41e73991a9f34578886b659a08a29b98%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_490fafbd65474605be447a6a0b6a8c70%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5873395516784%2C9.70869541168213%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_714ff84df6ac48ba82bf85813e4546a9%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bd14019dd61e4cc98e81ccb9bf7d36ce%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_bd14019dd61e4cc98e81ccb9bf7d36ce%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Opn%20Klint%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_714ff84df6ac48ba82bf85813e4546a9.setContent%28html_bd14019dd61e4cc98e81ccb9bf7d36ce%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_490fafbd65474605be447a6a0b6a8c70.bindPopup%28popup_714ff84df6ac48ba82bf85813e4546a9%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1e6544b97c874df28ee5d97f6714324a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5863301162117%2C9.69326734542847%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_38265b3f63df4540a7a6f36215f4e59a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_30d79c94ddbc49768d4b1319ea753f16%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_30d79c94ddbc49768d4b1319ea753f16%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Reepschl%C3%A4gerstra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_38265b3f63df4540a7a6f36215f4e59a.setContent%28html_30d79c94ddbc49768d4b1319ea753f16%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_1e6544b97c874df28ee5d97f6714324a.bindPopup%28popup_38265b3f63df4540a7a6f36215f4e59a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1650da21464647b5a0f019d0068db548%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5898009446568%2C9.69020962715149%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_36c4188232e04faa9a63e50481aad675%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_edd69d515b664ad8887c94591dd28564%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_edd69d515b664ad8887c94591dd28564%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Schlehdornweg%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_36c4188232e04faa9a63e50481aad675.setContent%28html_edd69d515b664ad8887c94591dd28564%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_1650da21464647b5a0f019d0068db548.bindPopup%28popup_36c4188232e04faa9a63e50481aad675%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_005a030221354ec78ca56fe97080ee38%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.59188443917404%2C9.724164381623268%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ff0000%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ff0000%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_002e3b30ea9447a2baf8ef054258d380%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2f25a6d265f445e0a2907d2cd8b6a0c2%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_2f25a6d265f445e0a2907d2cd8b6a0c2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Von-Suttner%20Stra%C3%9Fe%20Wedel%20Cluster%200%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_002e3b30ea9447a2baf8ef054258d380.setContent%28html_2f25a6d265f445e0a2907d2cd8b6a0c2%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_005a030221354ec78ca56fe97080ee38.bindPopup%28popup_002e3b30ea9447a2baf8ef054258d380%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_309a7475fc6f49129bf770c99a093c55%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5884908446434%2C9.70395349985231%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2822b2f94cf64a3da33575fad3291977%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_801606098e5341d2b20578f745ec8afb%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_801606098e5341d2b20578f745ec8afb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Wiedetwiete%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2822b2f94cf64a3da33575fad3291977.setContent%28html_801606098e5341d2b20578f745ec8afb%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_309a7475fc6f49129bf770c99a093c55.bindPopup%28popup_2822b2f94cf64a3da33575fad3291977%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_26fcf5aa6e3f403ba2ed7659773aa433%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5824138041649%2C9.71122829167579%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_23fad1822f3f40719ec547f4d52cb2f9%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fa86d4e0f79143f4a3fbe193c2e0805f%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_fa86d4e0f79143f4a3fbe193c2e0805f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Autal%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_23fad1822f3f40719ec547f4d52cb2f9.setContent%28html_fa86d4e0f79143f4a3fbe193c2e0805f%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_26fcf5aa6e3f403ba2ed7659773aa433.bindPopup%28popup_23fad1822f3f40719ec547f4d52cb2f9%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_91b14ddd9af049d6aa384192b01e854c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.575454069497%2C9.7262316942215%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_4655de9752ca41a9aa0dc37b1e306b72%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e93b69006aa949ebab2f527511744d7a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_e93b69006aa949ebab2f527511744d7a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Im%20Grund%20Wedel%20Cluster%203%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_4655de9752ca41a9aa0dc37b1e306b72.setContent%28html_e93b69006aa949ebab2f527511744d7a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_91b14ddd9af049d6aa384192b01e854c.bindPopup%28popup_4655de9752ca41a9aa0dc37b1e306b72%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_19a8c39916784c5eaf35ebdd8281cfe9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5695355602878%2C9.70255315303802%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_7906026eb43543448d4aef79a3c31208%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f25f93313f744c198a7d3f90b29d2875%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_f25f93313f744c198a7d3f90b29d2875%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Parna%C3%9Fstra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_7906026eb43543448d4aef79a3c31208.setContent%28html_f25f93313f744c198a7d3f90b29d2875%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_19a8c39916784c5eaf35ebdd8281cfe9.bindPopup%28popup_7906026eb43543448d4aef79a3c31208%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_30c792236e1547aeaf8958bf1e174fca%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5933545865536%2C9.71821457147598%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_ba907c754da247b1ae8cebc1d9ec2435%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c1ade56754324ea3b2eb7ee5f9eb66e5%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c1ade56754324ea3b2eb7ee5f9eb66e5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Rebhuhnweg%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_ba907c754da247b1ae8cebc1d9ec2435.setContent%28html_c1ade56754324ea3b2eb7ee5f9eb66e5%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_30c792236e1547aeaf8958bf1e174fca.bindPopup%28popup_ba907c754da247b1ae8cebc1d9ec2435%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_74d451189ae14dd58a3cbb16f3dec86e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.59039491694423%2C9.69691358503951%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_28faf720ffbb4acd9284b9773a595998%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_af99a33ac0364d58bf2eedfc7a60fcdc%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_af99a33ac0364d58bf2eedfc7a60fcdc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Appelboomtwiete%20Ecke%20Aastwiete%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_28faf720ffbb4acd9284b9773a595998.setContent%28html_af99a33ac0364d58bf2eedfc7a60fcdc%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_74d451189ae14dd58a3cbb16f3dec86e.bindPopup%28popup_28faf720ffbb4acd9284b9773a595998%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a88220b58e1848368776b9ba6cf0b963%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5754046192883%2C9.71914261579514%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_850b348ee4c740df9081ae42f9c3a297%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2a26760e937f457a8cb82077262c6ea5%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_2a26760e937f457a8cb82077262c6ea5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Tinsdaler%20Weg%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_850b348ee4c740df9081ae42f9c3a297.setContent%28html_2a26760e937f457a8cb82077262c6ea5%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a88220b58e1848368776b9ba6cf0b963.bindPopup%28popup_850b348ee4c740df9081ae42f9c3a297%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6308f01640f4452ea01501d2e79a7077%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5822166562335%2C9.70818042755127%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_9fe61e497e4344d3a5e0dd64edff1b87%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2ae35f6c77024036a346e2edc8336b8f%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_2ae35f6c77024036a346e2edc8336b8f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Theaterstra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_9fe61e497e4344d3a5e0dd64edff1b87.setContent%28html_2ae35f6c77024036a346e2edc8336b8f%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_6308f01640f4452ea01501d2e79a7077.bindPopup%28popup_9fe61e497e4344d3a5e0dd64edff1b87%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c74b7d1c168844df9c8d2d7431c47892%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B53.5937489838527%2C9.73056882619858%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22%23ffb360%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%23ffb360%22%2C%0A%20%20%22fillOpacity%22%3A%200.7%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a85d11c71d094812aef2dfc89f659ff6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_29db3abb41c744da88c4aaf3e299b739%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fb5e2dcf5e2a41c3a6b75233b374be3a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_fb5e2dcf5e2a41c3a6b75233b374be3a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3E%20Spielplatz%20Heinestra%C3%9Fe%20Wedel%20Cluster%204%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_29db3abb41c744da88c4aaf3e299b739.setContent%28html_fb5e2dcf5e2a41c3a6b75233b374be3a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_c74b7d1c168844df9c8d2d7431c47892.bindPopup%28popup_29db3abb41c744da88c4aaf3e299b739%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



#### Analyzing the playground equipment-based groupings:


```python
'''
These are mostly some pretty good playgrounds. Probably good for an hour-long trip.
'''
final_df.loc[final_df['Cluster Labels'] == 0, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Haselweg Wedel</td>
      <td>53.591291</td>
      <td>Schöner Spielplatz mit angegliederter kleiner...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Meisenweg Wedel</td>
      <td>53.594306</td>
      <td>Großer Spielplatz mit viel Wiese. Die Spielge...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Mühlenweg Wedel</td>
      <td>53.581066</td>
      <td>Schön gestalteter Spielplatz am Mühlenweg.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Ginsterweg Wedel</td>
      <td>53.573638</td>
      <td>Der Spielplatz ist auf mehrere Ebenen in eine...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hans-Böckler Platz Wedel</td>
      <td>53.568860</td>
      <td>Der Spielplatz ist zur Straße hin mit einem Z...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pulverstraße Wedel</td>
      <td>53.571205</td>
      <td>Dieser mittelgroße Spielplatz befindet sich n...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Alter Zirkusplatz Wedel</td>
      <td>53.575596</td>
      <td>Versteckter Spielplatz mit schattigen Ecken u...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Brombeerweg Wedel</td>
      <td>53.574119</td>
      <td>Schöner kleiner Spielplatz unter Bäumen&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Croningstraße Wedel</td>
      <td>53.581910</td>
      <td>Dieser Spielplatz nur von der Croningstraße e...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Ernst-Thälmann-Weg Wedel</td>
      <td>53.588167</td>
      <td>Eingeschlossen von Häusern liegt hier ein ruh...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Rosengarten Wedel</td>
      <td>53.581053</td>
      <td>Dieser Spielplatz liegt am Rosengarten, nahe ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Schwartenseekamp Wedel</td>
      <td>53.598917</td>
      <td>Diesen Spielplatz haben wir noch nicht besuch...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Wacholderstraße Wedel</td>
      <td>53.590775</td>
      <td>Spielplatz mit Schwerpunkt Sandspiele.&lt;br/&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Kronskamp Wedel</td>
      <td>53.580795</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Bürgerpark Wedel</td>
      <td>53.584817</td>
      <td>Kleiner Spielplatz im Bürgerpark.</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Egenbüttelweg Wedel</td>
      <td>53.591180</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Von-Suttner Straße Wedel</td>
      <td>53.591884</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
This is an exceptional playground, the kind that parents send pictures of to other people. 
In a class of its own so this clustering of one makes sense. There's probably even some equipment that's not
listed here.
'''
final_df.loc[final_df['Cluster Labels'] == 1, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Waldspielplatz Moorwegsiedlung Wedel</td>
      <td>53.592631</td>
      <td>Großer Spielplatz im Wald. Viel Wiese.</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
One nice result is that the clustering algorithm has placed all the playgrounds with water features (pumps, troughs, 
water wheels, etc.) into this category. They tend to have less of the standard playground equipment, and are more
just focused on water play.
'''
final_df.loc[final_df['Cluster Labels'] == 2, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>53.591441</td>
      <td>Der Spielplatz macht einen herausragenden Eind...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Rotdornstraße Wedel</td>
      <td>53.591067</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Anne-Frank-Weg Wedel</td>
      <td>53.588950</td>
      <td>Spielplatz mit Matschanlage (also die Ersatzk...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Gärtnerstraße Wedel</td>
      <td>53.585607</td>
      <td>Dieser Spielplatz wurde seit meinem letzten B...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
Also good playgrounds, definitely good for an hour or so of children's play. 
One defining difference is that these playgrounds all include football/soccer fields.
'''
final_df.loc[final_df['Cluster Labels'] == 3, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Ansgariusweg Wedel</td>
      <td>53.587196</td>
      <td>An der Zufahrt zum Fährmannssand liegt dieser...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Geesthang Wedel</td>
      <td>53.589786</td>
      <td>Dieser Spielplatz befindet sich am Ende der H...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Heinrich-Schacht-Straße Wedel</td>
      <td>53.580021</td>
      <td>Ein größerer Spielplatz. Bemerkenswert neben ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Strandbad Wedel</td>
      <td>53.570948</td>
      <td>Großer Spielplatz:&lt;br/&gt;Wegen des Wassers in d...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hellgrund Wedel</td>
      <td>53.566977</td>
      <td>Versteckt im Tal in direkter Nähe zum Vattenf...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Opn Klint Wedel</td>
      <td>53.587340</td>
      <td>Dieser Spielplatz ist zum Teil öffentlch, zum...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Im Grund Wedel</td>
      <td>53.575454</td>
      <td>Zunächst findet man hier den Bolzplatz, dahin...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
Some of these playgrounds have less going on and are probably more neighborhood playgrounds than those
worth driving to. It also has the playgrounds without rating or poorly listed equipment.

There are a couple ways to interpret these. Since the data is crowdsourced, it could be that these just have not had
much data provided to the webpage. In that regard, they're wildcards. So they may be worth exploring and could be
surprising. However, they could also be playgrounds with limited equipment available and not much fun for the kids.
For the ones I'm familiar with, the latter explanation is most often the case. One has an oddly high rating for not
having much equipment available.
'''

final_df.loc[final_df['Cluster Labels'] == 4, final_df.columns[[1] + [2] + list(range(5, final_df.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spielplatz Hamburger Yachthafen Wedel</td>
      <td>53.574397</td>
      <td>neuer, riesiger toller spielplatz, muss man h...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Altstadtschule Wedel</td>
      <td>53.582625</td>
      <td>Dieser Spielplatz liegt auf dem Schulhof der ...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Lindenstraße Wedel</td>
      <td>53.578413</td>
      <td>Ein langezogener Spielplatz mit insgesamt vie...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pferdekoppel Wedel</td>
      <td>53.588979</td>
      <td>Dieser Spielplatz mit schöner Spielburg liegt...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Pinneberger Straße Wedel</td>
      <td>53.585592</td>
      <td>Abgegrenzt vom Obstbaumweg an der Rückseite u...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Vogt-Körner Straße Wedel</td>
      <td>53.575123</td>
      <td>Dieser kleine Spielplatz liegt versteckt hint...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Appelboomtwiete Ecke Steinberg Wedel</td>
      <td>53.588835</td>
      <td>Diesen Spielplatz haben wir noch nicht besuch...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Albert-Schweizer Schule Wedel</td>
      <td>53.571721</td>
      <td>Dieser Spielplatz befindet sich auf dem Gelän...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Elbstraße Wedel</td>
      <td>53.569940</td>
      <td>Kleiner Spielplatz.&lt;br/&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Gerhart-Hauptmann Straße Wedel</td>
      <td>53.591999</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Hainbuchenweg Wedel</td>
      <td>53.589938</td>
      <td>Spielplatz eher für etwas ältere Kinder.&lt;br/&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Klintkamp Wedel</td>
      <td>53.591943</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Reepschlägerstraße Wedel</td>
      <td>53.586330</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Schlehdornweg Wedel</td>
      <td>53.589801</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Wiedetwiete Wedel</td>
      <td>53.588491</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Autal Wedel</td>
      <td>53.582414</td>
      <td>Sehr kleiner Spielplatz. Gelegen nahe an der ...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Parnaßstraße Wedel</td>
      <td>53.569536</td>
      <td>Kleiner Spielplatz im Parnaßpark, dem die Gra...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Rebhuhnweg Wedel</td>
      <td>53.593355</td>
      <td>Kleiner Spielplatz&lt;br/&gt;</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Appelboomtwiete Ecke Aastwiete Wedel</td>
      <td>53.590395</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Tinsdaler Weg Wedel</td>
      <td>53.575405</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Theaterstraße Wedel</td>
      <td>53.582217</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Spielplatz Heinestraße Wedel</td>
      <td>53.593749</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Part IID: Finding the playgrounds that are near fast food restaurants, icecream shops, etc. <a name="bonus2"></a>
Sometimes it's more important to find a playground with a particular venue nearby. Here's some of those sets.

#### Find the playgrounds with 'fast food restaurants' nearby:


```python
'''
Listing the playgrounds but also the restaurant names. 
Kids gotta eat and sometimes playground adventures go longer than planned. 
If it seems like one of those days, maybe it's best to go to one of these playgrounds.
'''
Playground_venues[Playground_venues['Venue Category']=='Fast Food Restaurant']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>Playground Latitude</th>
      <th>Playground Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>Spielplatz Mühlenweg Wedel</td>
      <td>53.581066</td>
      <td>9.710192</td>
      <td>Hähnchengrill Wedel</td>
      <td>53.579899</td>
      <td>9.703757</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Spielplatz Altstadtschule Wedel</td>
      <td>53.582625</td>
      <td>9.699267</td>
      <td>Hähnchengrill Wedel</td>
      <td>53.579899</td>
      <td>9.703757</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Spielplatz Croningstraße Wedel</td>
      <td>53.581910</td>
      <td>9.723378</td>
      <td>Burger King</td>
      <td>53.583856</td>
      <td>9.726297</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Spielplatz Croningstraße Wedel</td>
      <td>53.581910</td>
      <td>9.723378</td>
      <td>McDonald's</td>
      <td>53.583530</td>
      <td>9.723654</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>123</th>
      <td>Spielplatz Heinrich-Schacht-Straße Wedel</td>
      <td>53.580021</td>
      <td>9.724880</td>
      <td>Burger King</td>
      <td>53.583856</td>
      <td>9.726297</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Spielplatz Heinrich-Schacht-Straße Wedel</td>
      <td>53.580021</td>
      <td>9.724880</td>
      <td>McDonald's</td>
      <td>53.583530</td>
      <td>9.723654</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Spielplatz Rosengarten Wedel</td>
      <td>53.581053</td>
      <td>9.705455</td>
      <td>Hähnchengrill Wedel</td>
      <td>53.579899</td>
      <td>9.703757</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Spielplatz Kronskamp Wedel</td>
      <td>53.580795</td>
      <td>9.722074</td>
      <td>Burger King</td>
      <td>53.583856</td>
      <td>9.726297</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Spielplatz Kronskamp Wedel</td>
      <td>53.580795</td>
      <td>9.722074</td>
      <td>McDonald's</td>
      <td>53.583530</td>
      <td>9.723654</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>300</th>
      <td>Spielplatz Theaterstraße Wedel</td>
      <td>53.582217</td>
      <td>9.708180</td>
      <td>Hähnchengrill Wedel</td>
      <td>53.579899</td>
      <td>9.703757</td>
      <td>Fast Food Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



#### Find the playgrounds near venues that service icecream:


```python
'''
Search for 'Eis' in the business title since it doesn't have it's own venue category. 
Note 'Eis' is German for 'icecream'. There is another restaurant in the vilage that has a walk-up icecream window,
but 'eis' isn't in the name. One option would be to vastly increase the number of Foursquare API calls and see if
we can check menus for icecream.
'''
Playground_venues[Playground_venues['Venue'].str.contains('Eis')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>Playground Latitude</th>
      <th>Playground Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>Spielplatz Alter Zirkusplatz Wedel</td>
      <td>53.575596</td>
      <td>9.710723</td>
      <td>Eiscafé Venezia</td>
      <td>53.577454</td>
      <td>9.70535</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>154</th>
      <td>Spielplatz Rosengarten Wedel</td>
      <td>53.581053</td>
      <td>9.705455</td>
      <td>Eiscafé Venezia</td>
      <td>53.577454</td>
      <td>9.70535</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>182</th>
      <td>Spielplatz Vogt-Körner Straße Wedel</td>
      <td>53.575123</td>
      <td>9.705176</td>
      <td>Eiscafé Venezia</td>
      <td>53.577454</td>
      <td>9.70535</td>
      <td>Café</td>
    </tr>
  </tbody>
</table>
</div>



#### Find the playgrounds that list water features:


```python
'''
These generally include various pumps, channels, waterwheels, etc. 
Very popular and useful in the summer. 
Unfortunately, the playgrounds where water features were noted does not correspond to the playgrounds 
near the icecream shop in the prior list (in the case of the village of Wedel).
'''
final_df[final_df['water feature']==1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster Labels</th>
      <th>Playground</th>
      <th>a_lat</th>
      <th>a_long</th>
      <th>a_name_address</th>
      <th>a_description</th>
      <th>a_rating</th>
      <th>water feature</th>
      <th>sandpit</th>
      <th>cable car</th>
      <th>playhouse</th>
      <th>tree house</th>
      <th>slide</th>
      <th>swing</th>
      <th>climbing features</th>
      <th>sledding hill</th>
      <th>football field</th>
      <th>seesaw</th>
      <th>basketball</th>
      <th>nest swing</th>
      <th>swings</th>
      <th>turntable</th>
      <th>carousel</th>
      <th>table tennis</th>
      <th>trampoline</th>
      <th>railroad</th>
      <th>tractor</th>
      <th>excavator</th>
      <th>climbing tower</th>
      <th>tunnel</th>
      <th>spring board</th>
      <th>blancing boards</th>
      <th>toilets</th>
      <th>bicycle stand</th>
      <th>total equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Spielplatz Wasserspielplatz Haus am See Wedel</td>
      <td>53.591441</td>
      <td>9.705530</td>
      <td>Spielplatz Wasserspielplatz Haus am See in Wed...</td>
      <td>Der Spielplatz macht einen herausragenden Eind...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Spielplatz Rotdornstraße Wedel</td>
      <td>53.591067</td>
      <td>9.688362</td>
      <td>Besonderheiten laut der Liste aus Wedel in Zah...</td>
      <td>Besonderheiten laut der Liste aus Wedel in Za...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Spielplatz Anne-Frank-Weg Wedel</td>
      <td>53.588950</td>
      <td>9.693766</td>
      <td>Spielplatz mit Matschanlage (also die Ersatzkl...</td>
      <td>Spielplatz mit Matschanlage (also die Ersatzk...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Spielplatz Gärtnerstraße Wedel</td>
      <td>53.585607</td>
      <td>9.697387</td>
      <td>Dieser Spielplatz wurde seit meinem letzten Be...</td>
      <td>Dieser Spielplatz wurde seit meinem letzten B...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



#### Find the playgrounds with supermarkets nearby:


```python
'''
Gotta shop sometimes, might as well do a shopping-playground trip and save time.
I'm partial to Netto, so searching based on that brand of supermarket.
'''
Playground_venues[(Playground_venues['Venue Category']=='Supermarket') 
                  & (Playground_venues['Venue'].str.contains("Netto"))]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playground</th>
      <th>Playground Latitude</th>
      <th>Playground Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>Spielplatz Hans-Böckler Platz Wedel</td>
      <td>53.568860</td>
      <td>9.714913</td>
      <td>Netto Marken-Discount</td>
      <td>53.569410</td>
      <td>9.713800</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Spielplatz Pulverstraße Wedel</td>
      <td>53.571205</td>
      <td>9.719400</td>
      <td>Netto Marken-Discount</td>
      <td>53.569410</td>
      <td>9.713800</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Spielplatz Anne-Frank-Weg Wedel</td>
      <td>53.588950</td>
      <td>9.693766</td>
      <td>Netto Marken-Discount</td>
      <td>53.585539</td>
      <td>9.691813</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Spielplatz Ansgariusweg Wedel</td>
      <td>53.587196</td>
      <td>9.686224</td>
      <td>Netto Marken-Discount</td>
      <td>53.585539</td>
      <td>9.691813</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Spielplatz Gärtnerstraße Wedel</td>
      <td>53.585607</td>
      <td>9.697387</td>
      <td>Netto Marken-Discount</td>
      <td>53.585539</td>
      <td>9.691813</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Spielplatz Ernst-Thälmann-Weg Wedel</td>
      <td>53.588167</td>
      <td>9.693680</td>
      <td>Netto Marken-Discount</td>
      <td>53.585539</td>
      <td>9.691813</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>215</th>
      <td>Spielplatz Bürgerpark Wedel</td>
      <td>53.584817</td>
      <td>9.692506</td>
      <td>Netto Marken-Discount</td>
      <td>53.585539</td>
      <td>9.691813</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Spielplatz Elbstraße Wedel</td>
      <td>53.569940</td>
      <td>9.711303</td>
      <td>Netto Marken-Discount</td>
      <td>53.569410</td>
      <td>9.713800</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>249</th>
      <td>Spielplatz Reepschlägerstraße Wedel</td>
      <td>53.586330</td>
      <td>9.693267</td>
      <td>Netto Marken-Discount</td>
      <td>53.585539</td>
      <td>9.691813</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>258</th>
      <td>Spielplatz Schlehdornweg Wedel</td>
      <td>53.589801</td>
      <td>9.690210</td>
      <td>Netto Marken-Discount</td>
      <td>53.585539</td>
      <td>9.691813</td>
      <td>Supermarket</td>
    </tr>
  </tbody>
</table>
</div>



# Discussion and concluding remarks <a name="discussionandconclusion"></a>

This section concludes the report with a discussion of the results and some concluding remarks.
<br />
<br />
The clustering methods used in this study appear to work well at dividing the playgrounds in the village of interest into groups. From experience with the village, clustering based on nearby venues captures differences in the village neighborhoods well. In particular, it is not ex ante apparent that the neighborhoods of clusters zero and one would fall into different clusters. Yet, the neighborhoods do have a different feel to them in real life. This exercise reveals that one of the sources of that difference is that different sorts of venues are concentrated in each. So too with clusters two and four which are in less-dense areas of the village (cluster three is also in an isolated corner on the Elbe beach).
<br />
<br />
When analyzing the data again based on clustering by playground characteristics, patterns again emerge. Many of the
playgrounds in the village are good and fairly consistent and these have been grouped together. The really exceptional is placed in a cluster of its own. Then, playgrounds with football/soccer fields are also grouped well, as are
those with water features. The playgrounds with less features and less information are sort of grouped together - those
that we wouldn't want to show up to expecting the children to enjoy for an hour. These are the ones we'd want to have a 
backup plan for. Overall, I'd say the clustering exercise has worked well.
<br />
<br />
Finally, sometimes it's about have a lot of shops nearby or specific the equipment on the playground. But other times it's 
about having a specific business or business category nearby. Using the playgrounds and Foursquare data results in a few
relevant lists. We now have lists of the playgrounds near fast food, icecream, and one of the supermarkets readily
available.
<br />
<br />
I think there is more that can be done with this sort of database. For instance, we could calculate the distance between each playground and the venues. This might be useful since walking with children can be difficult. I have also considered adding a visual analysis. For instance, I could use the coordinates to retrieve some satellite imagery, and then use some sort of computer vision approach to check how much shade is available. That could be an interesting and useful addition.  Parking and other features would be interesting to retrieve too. Anyway, linking the Foursquare API to the data scraped from the community crowd-source spielplatznet website has led to some interesting and useful insights.
