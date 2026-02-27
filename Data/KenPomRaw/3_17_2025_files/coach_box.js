$(function() {
    var availableTags = [
        "Abilene Christian","Air Force","Akron","Alabama","Alabama A&M","Alabama St.","Albany","Alcorn St.","American","Appalachian St.",
        "Arizona","Arizona St.","Arkansas","Little Rock","Arkansas Pine Bluff","Arkansas St.","Army","Auburn","Austin Peay",
        "Ball St.","Baylor","Belmont","Bellarmine","Bethune Cookman","Binghamton","Boise St.","Boston College","Boston University","Bowling Green",
        "Bradley","Brown","Bryant","Bucknell","Buffalo","Butler","BYU","Cal Baptist","Cal Poly","Cal St. Bakersfield","Cal St. Fullerton",
        "CSUN","California","Campbell","Canisius","Central Arkansas","Central Connecticut","Central Michigan","Charleston",
        "Charleston Southern","Charlotte","Chattanooga","Chicago St.","Cincinnati","Clemson","Cleveland St.","Coastal Carolina",
        "Colgate","Colorado","Colorado St.","Columbia","Connecticut","Coppin St.","Cornell","Creighton",
        "Dartmouth","Davidson","Dayton","Delaware","Delaware St.","Denver","DePaul","Detroit Mercy","Utah Tech","Drake","Drexel","Duke","Duquesne",
        "East Carolina","East Tennessee St.","Eastern Illinois","Eastern Kentucky","Eastern Michigan","Eastern Washington","Elon",
        "Evansville","Fairfield","Fairleigh Dickinson","FIU","Florida","Florida A&M","Florida Atlantic","Florida Gulf Coast","Florida St.",
        "Fordham","Fresno St.","Furman","Gardner Webb","George Mason","George Washington","Georgetown","Georgia","Georgia Southern",
        "Georgia St.","Georgia Tech","Gonzaga","Grambling St.","Grand Canyon","Green Bay","Hampton","Harvard","Hawaii",
        "High Point","Hofstra","Holy Cross","Houston","Houston Christian","Howard","Idaho","Idaho St.","Illinois","Illinois Chicago",
        "Illinois St.","Incarnate Word","Indiana","Indiana St.","Iona","Iowa","Iowa St.","IU Indy","Jackson St.","Jacksonville",
        "Jacksonville St.","James Madison","Kansas","Kansas St.","Kennesaw St.","Kent St.","Kentucky","La Salle","Lafayette","Lamar",
        "Lehigh","Le Moyne","Liberty","Lindenwood","Lipscomb","LIU","Long Beach St.","Longwood","Louisiana","Louisiana Monroe","Louisiana Tech",
        "Louisville","Loyola Chicago","Loyola Marymount","Loyola MD","LSU","Maine","Manhattan","Marist","Marquette","Marshall","Maryland",
        "Maryland Eastern Shore","Massachusetts","McNeese","Memphis","Mercer","Merrimack","Miami FL","Miami OH","Michigan","Michigan St.",
        "Middle Tennessee","Milwaukee","Minnesota","Mississippi","Mississippi St.","Mississippi Valley St.","Missouri","Missouri St.",
        "Monmouth","Montana","Montana St.","Morehead St.","Morgan St.","Mount St. Mary's","Murray St.","Navy","Nebraska","Nebraska Omaha",
        "Nevada","New Hampshire","New Mexico","New Mexico St.","New Orleans","Niagara","Nicholls","NJIT","Norfolk St.","North Alabama",
        "North Carolina","North Carolina A&T","North Carolina Central","N.C. State","North Dakota","North Dakota St.",
        "North Florida","North Texas","Northeastern","Northern Arizona","Northern Colorado","Northern Illinois","Northern Iowa",
        "Northern Kentucky","Northwestern","Northwestern St.","Notre Dame","Oakland","Ohio","Ohio St.","Oklahoma","Oklahoma St.",
        "Old Dominion","Oral Roberts","Oregon","Oregon St.","Pacific","Penn","Penn St.","Pepperdine","Pittsburgh","Portland",
        "Portland St.","Prairie View A&M","Presbyterian","Princeton","Providence","Purdue","Purdue Fort Wayne","Queens","Quinnipiac","Radford",
        "Rhode Island","Rice","Richmond","Rider","Robert Morris","Rutgers","Sacramento St.","Sacred Heart","Saint Joseph's","Saint Louis",
        "Saint Mary's","Saint Peter's","Sam Houston St.","Samford","San Diego","San Diego St.","San Francisco","San Jose St.",
        "Santa Clara","Savannah St.","Seattle","Seton Hall","Siena","SIUE","SMU","South Alabama","South Carolina",
        "South Carolina St.","South Dakota","South Dakota St.","South Florida","Southeast Missouri","Southeastern Louisiana","St. Thomas",
        "Southern","Southern Indiana","Southern Illinois","Southern Miss","Southern Utah","St. Bonaventure","Saint Francis","St. John's",
        "Stanford","Stephen F. Austin","Stetson","Stonehill","Stony Brook","Syracuse","Tarleton St.","TCU","Temple","Tennessee","Tennessee Martin","Tennessee St.",
        "Tennessee Tech","Texas","Texas A&M","East Texas A&M","Texas A&M Corpus Chris","UT Rio Grande Valley","Texas Southern","Texas St.","Texas Tech",
        "The Citadel","Toledo","Towson","Troy","Tulane","Tulsa","UAB","UC Davis","UC Irvine","UC Riverside","UC Santa Barbara","UC San Diego",
        "UCF","UCLA","UMass Lowell","UMBC","Kansas City","UNC Asheville","UNC Greensboro","UNC Wilmington","UNLV","USC","USC Upstate","UT Arlington",
        "Utah","Utah St.","Utah Valley","UTEP","UTSA","Valparaiso","Vanderbilt","VCU","Vermont","Villanova","Virginia","Virginia Tech",
        "VMI","Wagner","Wake Forest","Washington","Washington St.","Weber St.","West Virginia","Western Carolina","Western Illinois",
        "Western Kentucky","Western Michigan","Wichita St.","William & Mary","Winthrop","Wisconsin","Wofford","Wright St.","Wyoming",
        "Xavier","Yale","Youngstown St.","Mercyhurst","West Georgia"
    ];
    $( "#teams" ).autocomplete({
        minLength: 2,
        source: availableTags,
        select: function(event,ui) {
            location.href = '/team.php?team=' + ui.item.value.replace("&","%26");
        }

    });
});
$(function() {
    var availableTags = ["Champions","Top Offenses","Top Defenses","Anthony Grant","Anthony Latina","Anthony Solomon","Anthony Stewart","Archie Miller","Armond Hill","Art Perry","Ashley Howard","Austin Claunch","Avery Johnson","BJ Hill","Bacari Alexander","Baker Dunleavy","Barclay Radebaugh","Barret Peery","Barry Collier","Barry Hinson","Barry Rohrssen","Bart Bellairs","Bart Lundy",
        "Bashir Mason","Ben Betts","Ben Braun","Ben Howland","Ben Jacobson","Ben Jobe","Ben Johnson","Benjy Taylor","Bennie Seltzer","Benny Moss","Bill Bayno","Bill Carmody","Bill Coen","Bill Courtney","Bill Dooley","Bill Evans","Bill Foster","Bill Frieder","Bill Grier","Bill Guthridge",
        "Bill Herrion","Bill Hodges","Bill Jones","Bill Musselman","Bill Raynor","Bill Self","Billy Donlon","Billy Donovan","Billy Gillispie","Billy Hahn","Billy Kennedy","Billy Lange","Billy Lee","Billy Taylor","Billy Tubbs","Billy Wright","Blaine Taylor","Bo Ellis","Bo Ryan","Bob Bender",
        "Bob Beyer","Bob Burton","Bob Donewald","Bob Hawking","Bob Hill","Bob Hoffman","Bob Huggins","Bob Knight","Bob Leckie","Bob Marlin","Bob McKillop","Bob Nash","Bob Richey","Bob Sundvold","Bob Thomason","Bob Walsh","Bob Weltlich","Bob Wenzel","Bob Williams","Bobby Braswell",
        "Bobby Collins","Bobby Cremins","Bobby Gonzalez","Bobby Hurley","Bobby Hussey","Bobby Jones","Bobby Lutz","Bobby Washington","Brad Brownell","Brad Greenberg","Brad Holland","Brad Huse","Brad Korn","Brad Soderberg","Brad Stevens","Brad Underwood","Brandon Laird","Brandon Miller","Bret Campbell","Brett Nelson",
        "Brett Reed","Brette Tanner","Brian Barone","Brian Burg","Brian Dutcher","Brian Earl","Brian Ellerbe","Brian Fish","Brian Gregory","Brian Hammel","Brian Jones","Brian Katz","Brian Kennedy","Brian Nash","Brian Wardle","Brooks Thompson","Bruce Pearl","Bruce Weber","Bruiser Flint","Bryan Mullins",
        "Bryce Drew","Bucky McMillan","Buster Harvey","Butch Beard","Buzz Peterson","Buzz Williams","Byron Rimm II","Byron Samuels","Byron Smith","C.B. McGrath","Cal Luther","Cameron Dollar","Carm Maciariello","Carson Cunningham","Carter Wilson","Casey Alexander","Charles Bradley","Charles Ramsey","Charlie Coles","Charlie Spoonhour",
        "Charlie Woollum","Charlton Young","Chico Potts","Chris Beard","Chris Casey","Chris Collins","Chris Fuller","Chris Holtmann","Chris Jans","Chris Knoche","Chris Lowery","Chris Mack","Chris Mooney","Chris Mullin","Chris Ogden","Chris Victor","Chris Walker","Chuck Driesell","Chuck Martin","Clarence Finley",
        "Clayton Bates","Clem Haskins","Clemon Johnson","Cliff Ellis","Cliff Warren","Clifford Reed Jr","Clyde Drexler","Corey Williams","Corliss Williamson","Craig Esherick","Craig Neal","Craig Rasmuson","Craig Robinson","Craig Smith","Cuonzo Martin","Curtis Hunter","Cy Alexander","Dale Brown","Dale Layer","Damon Stoudamire",
        "Dan D'Antoni","Dan Dakich","Dan Earl","Dan Engelstad","Dan Fitzgerald","Dan Hipsher","Dan Hurley","Dan Kenney","Dan Leibovitz","Dan Majerle","Dan McHale","Dan Monson","Dan Muller","Dan Peters","Dana Altman","Dana Ford","Dane Fife","Dane Fischer","Danny Kaspar","Danny Manning",
        "Danny Nee","Danny Sprinkle","Darelle Porter","Darian DeVries","Darrell Hawkins","Darrell Walker","Darrin Horn","Darris Nichols","Dave Balza","Dave Bezold","Dave Bike","Dave Bliss","Dave Bollwinkel","Dave Boots","Dave Calloway","Dave Dickerson","Dave Faucher","Dave Leitao","Dave Loos","Dave Magarity",
        "Dave McLaughlin","Dave Odom","Dave Paulsen","Dave Pilipovich","Dave Rice","Dave Richman","Dave Rose","Dave Simmons","Dave Wojcik","Davey Whitney","David Carter","David Cox","David Farrar","David Henderson","David Hobbs","David Kiefer","David Padgett","David Patrick","David Riley","David Spencer",
        "Dean Demopoulos","Dean Keener","Dean Smith","Deane Martin","Dedrique Taylor","Delray Brooks","Dennis Cutts","Dennis Felton","Dennis Gates","Dennis Nutt","Dennis Wolff","Denny Crum","Dereck Whittenburg","Derek Allister","Derek Kellogg","Derek Thomas","Derek Thompson","Derek Waugh","Derrin Hansen","Desmond Oliver",
        "Dick Bennett","Dick Davey","Dick Fick","Dick Hunsaker","Dick Kuchen","Dickey Nutt","Dino Gaudio","Doc Sadler","Don DeVoe","Don Friday","Don Harnum","Don Haskins","Don Holst","Don Maestri","Don Newman","Don Verlin","Donnie Jones","Donnie Marsh","Donnie Tyndall","Donny Daniels",
        "Donte Jackson","Donyell Marshall","Doug Davalos","Doug Noll","Doug Oliver","Doug Wojcik","Drew Valentine","Duane Reboul","Duggar Baucom","Dustin Kerns","Dusty May","Dwayne Killings","Dwight Freeman","Dylan Howard","Earl Grant","Ed Conroy","Ed Cooley","Ed Daniels Jr","Ed DeChellis","Ed Schilling",
        "Eddie Biedenbach","Eddie Fogler","Eddie Jordan","Eddie McCarter","Eddie Payne","Eddie Sutton","Edward Joyner Jr","Eldon Miller","Elwood Plummer","Emmett Davis","Eran Ganot","Eric Henderson","Eric Konkol","Eric Musselman","Eric Olen","Eric Reveno","Eric Skeeters","Ernie Kent","Ernie Nestor","Ernie Zeigler",
        "Eugene Harris","Fran Dunphy","Fran Fraschilla","Fran McCaffery","Fran O'Hanlon","Frank Dobbs","Frank Haith","Frank Harrell","Frank Martin","Frank Sullivan","Frankie Allen","Fred Hill","Fred Hoiberg","Fred Trenkle","G.G. Smith","Gale Catlett","Gary Garner","Gary Stewart","Gary Waters","Gary Williams",
        "Gene Cross","Gene Keady","Geno Ford","George Blaney","George Ivory","George Nessman","George Pfeifer","Gerald Gillion","Gib Arnold","Gil Jackson","Glen Miller","Glenn Braica","Grant McCasland","Gravelle Craig","Greg Gard","Greg Gary","Greg Graham","Greg Herenda","Greg Jackson","Greg Kampe",
        "Greg Lansing","Greg McDermott","Greg Paulus","Greg Vetrone","Greg White","Greg Young","Gregg Marshall","Gregg Nibert","Gregg Polinsky","Grey Giovanine","Griff Aldrich","Harold Blevins","Harry Miller","Heath Schroyer","Henry Bibby","Henry Dickerson","Herb Sendek","Herb Williams","Homer Drew","Horace Broadnax",
        "Houston Fancher","Howard Moore","Howie Dickenman","Hubert Davis","Hugh Durham","Isaac Brown","Isiah Thomas","J.D. Barnett","JP Piper","Jack Armstrong","Jack Bruen","Jack Grant","Jack Murphy","Jack Owens","Jack Perri","Jamal Brown","James Dickey","James Green","James Johnson","James Jones",
        "James Paul Casciano","James Whitford","Jamie Dixon","Jamion Christian","Jan van Breda Kolff","Jared Grasso","Jase Coburn","Jason Capel","Jason Crafton","Jason Gardner","Jason Hooten","Jason James","Jason Rabedeaux","Jason Shay","Jay John","Jay Joyner","Jay Ladner","Jay McAuley","Jay Smith","Jay Spoonhour",
        "Jay Wright","Jay Young","Jayson Gee","Jean Prioleau","Jeff Boals","Jeff Bower","Jeff Bzdelik","Jeff Capel II","Jeff Capel III","Jeff Jackson","Jeff Jones","Jeff Lebo","Jeff Linder","Jeff Meyer","Jeff Neubauer","Jeff Price","Jeff Reynolds","Jeff Ruland","Jeff Rutter","Jeff Schneider",
        "Jeff Wulbrun","Jeremy Ballard","Jerod Haase","Jerome Allen","Jerome Jenkins","Jerrod Calhoun","Jerry DeGregorio","Jerry Dunn","Jerry Eaves","Jerry Francis","Jerry Green","Jerry Hopkins","Jerry Pimm","Jerry Slocum","Jerry Stackhouse","Jerry Steele","Jerry Tarkanian","Jerry Wainwright","Jesse Agel","Jessie Evans",
        "Jim Baron","Jim Boeheim","Jim Boone","Jim Boylen","Jim Brown","Jim Calhoun","Jim Christian","Jim Crews","Jim Engles","Jim Ferry","Jim Fox","Jim Harrick","Jim Hayford","Jim Kerwin","Jim Larranaga","Jim Les","Jim Molinari","Jim O'Brien","Jim Phelan","Jim Platt",
        "Jim Saia","Jim Whitesell","Jim Wooldridge","Jim Yarbrough","Jimmy Allen","Jimmy Collins","Jimmy DuBose","Jimmy Lallathin","Jimmy Patsos","Jimmy Tillette","Jimmy Tubbs","Joby Wright","Joe Callero","Joe Cantafio","Joe Cravens","Joe DeSantis","Joe Dooley","Joe Gallo","Joe Golding","Joe Jones",
        "Joe Mihalich","Joe O'Brien","Joe Pasternack","Joe Scott","Joel Sobotka","Joey James","Joey Meyer","Joey Stiebing","John Aiken","John Becker","John Beilein","John Brady","John Brannen","John Calipari","John Chaney","John Cooper","John Dunne","John Gallagher","John Giannini","John Groce",
        "John Kresse","John Leonard","John Lyles","John MacLeod","John Masi","John Olive","John Pelphrey","John Phillips","John Robic","John Shulman","John Smith","John Thompson III","John Thompson","Johnny Dawkins","Johnny Jones","Johnny Tauer","Jon Coffman","Jon Harris","Jon Judkins","Jordan Mincy",
        "Joseph Price","Josh Pastner","Josh Schertz","Juan Dixon","Justin Gray","Justin Hutson","Juwan Howard","Kareem Richardson","Karl Hobbs","Keith Brown","Keith Dambrot","Keith Richard","Keith Walker","Kelvin Sampson","Ken Bone","Ken Burmeister","Ken McDonald","Kenny Blakeney","Keno Davis","Kermit Davis Jr",
        "Kerry Keating","Kerry Rupp","Kevin Baggett","Kevin Bannon","Kevin Billerman","Kevin Broadus","Kevin Bromley","Kevin Clark","Kevin Eastman","Kevin Johnson","Kevin Jones","Kevin Keatts","Kevin Kruger","Kevin McGeehan","Kevin McKenna","Kevin Nickelberry","Kevin O'Neill","Kevin Ollie","Kevin Stallings","Kevin Willard",
        "Kim Anderson","Kim English","King Rice","Kirk Earlywine","Kirk Saulny","Kirk Speraw","Kurt Kanaskie","Kyle Keller","Kyle Macy","Kyle Neptune","Kyle Perry","Kyle Smith","LaVall Jordan","Lafayette Stribling","Lamont Paris","Lamont Smith","Lance Irvin","Landon Bussie","Larry Brown","Larry Davis",
        "Larry Eustachy","Larry Farmer","Larry Finch","Larry Harrison","Larry Hunter","Larry Krystkowiak","Larry Lessett","Larry Reynolds","Larry Shyatt","Larry Smith","Larry Wright","LeVelle Moton","Lefty Driesell","Lennie Acuff","Lennox Forrester","Leon Rice","Leonard Drake","Leonard Hamilton","Leonard Perry","Levell Sanders",
        "Lew Hill","Lewis Jackson","Lewis Preston","Linc Darner","Lindsey Hunter","Lon Kruger","Lonnie Williams","Lorenzo Romar","Lou Henson","Louis Orr","Louis Rowe","Luke Yaklich","Lute Olson","Luther Riley","Mack McCarthy","Mark Adams","Mark Byington","Mark Few","Mark Fox","Mark Gottfried",
        "Mark Macon","Mark Madsen","Mark Montgomery","Mark Phelps","Mark Pope","Mark Price","Mark Prosser","Mark Schmidt","Mark Simons","Mark Slessinger","Mark Slonaker","Mark Turgeon","Martin Ingelsby","Marty Fletcher","Marty Simmons","Marty Wilson","Marvin Menzies","Matt Brady","Matt Brown","Matt Crenshaw",
        "Matt Doherty","Matt Figger","Matt Kilcullen","Matt Langel","Matt Lottich","Matt Matheny","Matt McCall","Matt McMahon","Matt Painter","Matthew Driscoll","Matthew Graves","Maurice Joseph","Max Good","Mel Coleman","Mel Hankinson","Melvin Watkins","Meredith Smith","Micah Shrewsberry","Michael Bernard","Michael Curry",
        "Michael Fly","Michael Grant","Michael Holton","Michael Hopkins","Michael Huger","Michael Hunt","Michael Perry","Michael White","Mick Cronin","Mick Durham","Mickey Clayton","Mike Adras","Mike Anderson","Mike Balado","Mike Boynton","Mike Brennan","Mike Brey","Mike Burns","Mike Calhoun","Mike Davis",
        "Mike Deane","Mike Dement","Mike Dunlap","Mike Dunleavy","Mike Garland","Mike Gillespie","Mike Gillian","Mike Heideman","Mike Hopkins","Mike Jarvis","Mike Jaskulski","Mike K. Jones","Mike R. Jones","Mike Krzyzewski","Mike LaPlante","Mike Lonergan","Mike MacDonald","Mike Magpayo","Mike Maker","Mike Martin","Mike McConathy",
        "Mike McLeese","Mike Miller","Mike Montgomery","Mike Morrell","Mike Rhoades","Mike Rice","Mike Schrage","Mike Sutton","Mike Vining","Mike Woodson","Mike Young","Milan Brown","Milton Barnes","Mitch Buonaguro","Mitch Henderson","Mo Cassara","Mo Williams","Monte Ross","Monte Towe","Montez Robinson",
        "Morris Scott","Murray Arnold","Murray Garvin","Murry Bartow","Nate James","Nate Oats","Nathan Davis","Neil Dougherty","Neil McCarthy","Nicholas McDevitt","Nick Macarchuk","Nick Robinson","Niko Medved","Nolan Richardson III","Nolan Richardson","Norm Roberts","Norm Stewart","Oliver Purnell","Orlando Antigua","Orlando Early",
        "Pat Baldwin","Pat Chambers","Pat Dennis","Pat Douglass","Pat Duquette","Pat Flannery","Pat Foster","Pat Harris","Pat Kelsey","Pat Kennedy","Pat Knight","Pat Sellers","Pat Skerry","Patrick Ewing","Patrick Sellers","Paul Aiello","Paul Biancardi","Paul Brazeau","Paul Cormier","Paul Graham",
        "Paul Hewitt","Paul Lusk","Paul Mills","Paul Sather","Paul Weir","Paul Westhead","Paul Westphal","Penny Collins","Penny Hardaway","Perry Clark","Perry Watson","Pete Gillen","Pete Strickland","Phil Cunningham","Phil Hopkins","Phil Johnson","Phil Martelli","Phil Mathews","Phil Rowe","Phillip Gary",
        "Porter Moser","Preston Spradlin","Quin Snyder","Quinton Ferrell","Ralph Willard","Rand Chappell","Randy Ayers","Randy Bennett","Randy Brown","Randy Dunton","Randy Monroe","Randy Peele","Randy Rahe","Randy Smithson","Randy Wiel","Rashon Burno","Ray Giacoletti","Ray Harper","Ray Lopes","Ray Martin",
        "Ray McCallum","Rees Johnson","Reggie Minton","Reggie Theus","Reggie Witherspoon","Rex Walters","Ric Cobb","Ricardo Patton","Rich Herrin","Rich Zvosec","Richard Barron","Richard Johnson","Richard Pitino","Richard Williams","Richie Riley","Rick Barnes","Rick Boyages","Rick Byrd","Rick Croy","Rick Majerus",
        "Rick Pitino","Rick Ray","Rick Samuels","Rick Scruggs","Rick Stansbury","Rickey Broussard","Ricky Blanton","Ricky Byrdsong","Ricky Duckett","Ricky Stokes","Riley Wallace","Ritchie McKay","Rob Chavez","Rob Ehsan","Rob Evans","Rob Flaska","Rob Jeter","Rob Judson","Rob Krimmel","Rob Lanier",
        "Rob Murphy","Rob Senderoff","Rob Spivery","Robbie Laing","Robert Burke","Robert Davenport","Robert Jones","Robert Lee","Robert McCullum","Robert Moreland","Rod Baker","Rod Barnes","Rod Jensen","Rodger Blind","Rodney Billups","Rodney Tention","Rodney Terry","Roger Reid","Rollie Massimino","Roman Banks",
        "Ron Abegglen","Ron Bradley","Ron Cottrell","Ron Everhart","Ron Ganulin","Ron Hunter","Ron Jirsa","Ron Mitchell","Ron Sanchez","Ron Shumate","Ron Verlin","Ronnie Arrow","Ronnie Courtney","Ronny Thompson","Roy Thomas","Roy Williams","Royce Waltman","Rudy Keeling","Russ Pennell","Russell Ellington",
        "Russell Turner","Ryan Looney","Ryan Marks","Ryan Odom","Ryan Ridder","Sal Mentesana","Sam Scholl","Sam Weaver","Samuel West","Saul Phillips","Scott Beeten","Scott Cherry","Scott Cross","Scott Davenport","Scott Drew","Scott Edgar","Scott Hicks","Scott Nagy","Scott Padgett","Scott Pera",
        "Scott Perry","Scott Sanderson","Scott Sutton","Scott Thompson","Sean Kearney","Sean Miller","Sean Sutton","Sean Woods","Sergio Rouco","Seth Greenberg","Shaheen Holloway","Shaka Smart","Shakey Rodriguez","Shane Burcar","Shantay Legans","Shawn Finney","Shawn Walker","Sherman Dillard","Sidney Green","Sidney Lowe",
        "Sidney Moncrief","Skip Prosser","Solomon Bozeman","Sonny Smith","Speedy Claxton","Speedy Morris","Stan Heath","Stan Johnson","Stan Joplin","Stan Morrison","Stan Waterman","Steve Aggers","Steve Alford","Steve Barnes","Steve Cleveland","Steve Donahue","Steve Fisher","Steve Forbes","Steve Hawkins","Steve Henson",
        "Steve Lappas","Steve Lavin","Steve Lutz","Steve Masiello","Steve McClain","Steve Merfeld","Steve Payne","Steve Pikiell","Steve Prohm","Steve Robinson","Steve Roccaforte","Steve Seymour","Steve Shields","Steve Shurina","Steve Smiley","Steve Wojciechowski","Stew Morrill","Sydney Johnson","T.J. Otzelberger","Tad Boyle",
        "Takayo Siddle","Tavaras Hardy","Ted Woodward","Terrence Johnson","Terry Carroll","Terry Dunn","Terry Porter","Terry Truax","Tevester Anderson","Thad Matta","Thomas Trotter","Tic Price","Tim Buckley","Tim Capstraw","Tim Carter","Tim Cluess","Tim Cohane","Tim Craft","Tim Duryea","Tim Floyd",
        "Tim Jankovich","Tim Miles","Tim O'Shea","Tim O'Toole","Tim Welsh","Tod Kowalczyk","Todd Bozeman","Todd Golden","Todd Howard","Todd Lee","Todd Lickliter","Todd Simon","Tom Abatemarco","Tom Asbury","Tom Brennan","Tom Conrad","Tom Crean","Tom Davis","Tom Green","Tom Herrion",
        "Tom Izzo","Tom McConnell","Tom Moore","Tom Parrotta","Tom Pecora","Tom Penders","Tom Richardson","Tom Schuberth","Tom Sullivan","Tommy Amaker","Tommy Dempsey","Tommy Lloyd","Tommy Vardeman","Tony Barbee","Tony Barone","Tony Benford","Tony Bennett","Tony Harvey","Tony Ingle","Tony Jasick",
        "Tony Madlock","Tony Pujol","Tony Shaver","Tony Sheals","Tony Stubblefield","Tracy Dildy","Travis DeCuire","Travis Ford","Travis Steele","Travis Williams","Trent Johnson","Tubby Smith","Tyler Geving","Van Holt","Vance Walberg","Vann Pettaway","Vic Trilli","Vonn Webb","Walter McCarty","Wayne Brent",
        "Wayne Morgan","Wayne Szoke","Wayne Tinkle","Wes Flanagan","Wes Miller","Wil Jones","Will Brown","Will Jones","Will Ryan","Will Wade","Willie Coward","Willie Hayes","Willis Wilson","Wimp Sanderson","Wyking Jones","Zac Claus","Zac Roman","Zach Spiker","Otis Hughley","Mike Lewis","Daniyal Robinson",
        "Matt McKillop","Jon Scheyer","Mike Schwartz","David Ragland","Tobin Anderson","Keith Urgo","Chris Caputo","Jonas Hayes","Ryan Pedon","Jerome Tang","Mike Jordan","Kyle Gerdeman","Rod Strickland","Talvin Hester","Kenny Payne","Chris Markwood","Chris Crutchfield","Greg Heiar","Phillip Shumpert",
        "Corey Gipson","Grant Leonard","Chris Gerlufsen","Erik Martin","Eric Peterson","Stan Gouard","Chris Kraus","Jaret Von Rosenberg","Andrew Wilson","Donald Copeland","Eric Duft","Dwayne Stephens","Andy Enfield",
        "Duane Simpkins","Bryan Hodgson","Kevin Kuwik","Phil Martelli Jr.","John Griffin","George Halcovage","Aaron Fearne","Larry Stewart","Brooks Savage","Jack Castleberry","Tony Skinn","Charlie Henry","Sundance Wicks","Alan Huss","Alex Pribble","Shane Heirman","Antoine Pettway","Mike McGarvey",
        "Nate Champion","Matt Logie","Tevon Saddler","Grant Billmeier","Ross Hodge","Rick Cabrera","Russell Springmann","Chris Mudge","Adrian Autry","Adam Fisher","Jim Shaw","K.T. Turner","Todd Phillips","Roger Powell","Josh Eilert","Chad Boudreau","Dwight Perry",
        "Kevin Young", "Mike DeGeorge", "Saah Nimley", "Scott Spinelli", "Jon Jaques", "Ben McCollum", "Dru Joyce", "Patrick Crarey", "John Jakus", "Jeremy Luther", "Doug Gottlieb", "Ivan Thomas", "Craig Doty", "Paul Corsaro", "Chris Acker", "Josh Loeffler", "Cornelius Jackson", "Gary Manchel", 
        "Jonathan Mattox", "Ben Fletcher", "Cleo Hill", "Donny Lind", "Stacy Hollowell", "Jake Diebler", "Mike K. Jones", "Dave Smart", "Michael Czepil", "Gerry McNamara", "Jeremy Shulman", "Marty Richter", "Kahil Fennell", "Dave Moore", "Hank Plona", "Clint Sargent", "Ethan Faulkner"
    ];
    $( "#coaches" ).autocomplete({
        minLength: 3,
        source: availableTags,
        select: function(event,ui) {
            location.href = '/history.php?c=' + ui.item.value;
        }
    });
});

// This adds 'placeholder' to the items listed in the jQuery .support object.
jQuery(function() {
    jQuery.support.placeholder = false;
    test = document.createElement('input');
    if('placeholder' in test) jQuery.support.placeholder = true;
});
/*!
 * jQuery Placeholder Plugin v2.1.3
 * https://github.com/mathiasbynens/jquery-placeholder
 *
 * Copyright 2011,2015 Mathias Bynens
 * Released under the MIT license
 */
(function(factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD
        define(['jquery'],factory);
    } else if (typeof module === 'object' && module.exports) {
        factory(require('jquery'));
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function($) {

    /****
     * Allows plugin behavior simulation in modern browsers for easier debugging.
     * When setting to true,use attribute "placeholder-x" rather than the usual "placeholder" in your inputs/textareas
     * i.e. <input type="text" placeholder-x="my placeholder text" />
     */
    var debugMode = false;

    // Opera Mini v7 doesn't support placeholder although its DOM seems to indicate so
    var isOperaMini = Object.prototype.toString.call(window.operamini) === '[object OperaMini]';
    var isInputSupported = 'placeholder' in document.createElement('input') && !isOperaMini && !debugMode;
    var isTextareaSupported = 'placeholder' in document.createElement('textarea') && !isOperaMini && !debugMode;
    var valHooks = $.valHooks;
    var propHooks = $.propHooks;
    var hooks;
    var placeholder;
    var settings = {};

    if (isInputSupported && isTextareaSupported) {

        placeholder = $.fn.placeholder = function() {
            return this;
        };

        placeholder.input = true;
        placeholder.textarea = true;

    } else {

        placeholder = $.fn.placeholder = function(options) {

            var defaults = {customClass: 'placeholder'};
            settings = $.extend({},defaults,options);

            return this.filter((isInputSupported ? 'textarea' : ':input') + '[' + (debugMode ? 'placeholder-x' : 'placeholder') + ']')
                .not('.'+settings.customClass)
                .not(':radio,:checkbox,:hidden')
                .bind({
                    'focus.placeholder': clearPlaceholder,
                    'blur.placeholder': setPlaceholder
                })
                .data('placeholder-enabled',true)
                .trigger('blur.placeholder');
        };

        placeholder.input = isInputSupported;
        placeholder.textarea = isTextareaSupported;

        hooks = {
            'get': function(element) {

                var $element = $(element);
                var $passwordInput = $element.data('placeholder-password');

                if ($passwordInput) {
                    return $passwordInput[0].value;
                }

                return $element.data('placeholder-enabled') && $element.hasClass(settings.customClass) ? '' : element.value;
            },
            'set': function(element,value) {

                var $element = $(element);
                var $replacement;
                var $passwordInput;

                if (value !== '') {

                    $replacement = $element.data('placeholder-textinput');
                    $passwordInput = $element.data('placeholder-password');

                    if ($replacement) {
                        clearPlaceholder.call($replacement[0],true,value) || (element.value = value);
                        $replacement[0].value = value;

                    } else if ($passwordInput) {
                        clearPlaceholder.call(element,true,value) || ($passwordInput[0].value = value);
                        element.value = value;
                    }
                }

                if (!$element.data('placeholder-enabled')) {
                    element.value = value;
                    return $element;
                }

                if (value === '') {

                    element.value = value;

                    // Setting the placeholder causes problems if the element continues to have focus.
                    if (element != safeActiveElement()) {
                        // We can't use `triggerHandler` here because of dummy text/password inputs :(
                        setPlaceholder.call(element);
                    }

                } else {

                    if ($element.hasClass(settings.customClass)) {
                        clearPlaceholder.call(element);
                    }

                    element.value = value;
                }
                // `set` can not return `undefined`; see http://jsapi.info/jquery/1.7.1/val#L2363
                return $element;
            }
        };

        if (!isInputSupported) {
            valHooks.input = hooks;
            propHooks.value = hooks;
        }

        if (!isTextareaSupported) {
            valHooks.textarea = hooks;
            propHooks.value = hooks;
        }

        $(function() {
            // Look for forms
            $(document).delegate('form','submit.placeholder',function() {

                // Clear the placeholder values so they don't get submitted
                var $inputs = $('.'+settings.customClass,this).each(function() {
                    clearPlaceholder.call(this,true,'');
                });

                setTimeout(function() {
                    $inputs.each(setPlaceholder);
                },10);
            });
        });

        // Clear placeholder values upon page reload
        $(window).bind('beforeunload.placeholder',function() {
            $('.'+settings.customClass).each(function() {
                this.value = '';
            });
        });
    }

    function args(elem) {
        // Return an object of element attributes
        var newAttrs = {};
        var rinlinejQuery = /^jQuery\d+$/;

        $.each(elem.attributes,function(i,attr) {
            if (attr.specified && !rinlinejQuery.test(attr.name)) {
                newAttrs[attr.name] = attr.value;
            }
        });

        return newAttrs;
    }

    function clearPlaceholder(event,value) {

        var input = this;
        var $input = $(this);

        if (input.value === $input.attr((debugMode ? 'placeholder-x' : 'placeholder')) && $input.hasClass(settings.customClass)) {

            input.value = '';
            $input.removeClass(settings.customClass);

            if ($input.data('placeholder-password')) {

                $input = $input.hide().nextAll('input[type="password"]:first').show().attr('id',$input.removeAttr('id').data('placeholder-id'));

                // If `clearPlaceholder` was called from `$.valHooks.input.set`
                if (event === true) {
                    $input[0].value = value;

                    return value;
                }

                $input.focus();

            } else {
                input == safeActiveElement() && input.select();
            }
        }
    }

    function setPlaceholder(event) {
        var $replacement;
        var input = this;
        var $input = $(this);
        var id = input.id;

        // If the placeholder is activated,triggering blur event (`$input.trigger('blur')`) should do nothing.
        if (event && event.type === 'blur' && $input.hasClass(settings.customClass)) {
            return;
        }

        if (input.value === '') {
            if (input.type === 'password') {
                if (!$input.data('placeholder-textinput')) {

                    try {
                        $replacement = $input.clone().prop({ 'type': 'text' });
                    } catch(e) {
                        $replacement = $('<input>').attr($.extend(args(this),{ 'type': 'text' }));
                    }

                    $replacement
                        .removeAttr('name')
                        .data({
                            'placeholder-enabled': true,
                            'placeholder-password': $input,
                            'placeholder-id': id
                        })
                        .bind('focus.placeholder',clearPlaceholder);

                    $input
                        .data({
                            'placeholder-textinput': $replacement,
                            'placeholder-id': id
                        })
                        .before($replacement);
                }

                input.value = '';
                $input = $input.removeAttr('id').hide().prevAll('input[type="text"]:first').attr('id',$input.data('placeholder-id')).show();

            } else {

                var $passwordInput = $input.data('placeholder-password');

                if ($passwordInput) {
                    $passwordInput[0].value = '';
                    $input.attr('id',$input.data('placeholder-id')).show().nextAll('input[type="password"]:last').hide().removeAttr('id');
                }
            }

            $input.addClass(settings.customClass);
            $input[0].value = $input.attr((debugMode ? 'placeholder-x' : 'placeholder'));

        } else {
            $input.removeClass(settings.customClass);
        }
    }

    function safeActiveElement() {
        // Avoid IE9 `document.activeElement` of death
        try {
            return document.activeElement;
        } catch (exception) {}
    }
}));
$('input,textarea').placeholder();

$(function() {
    $( 'th' ).tooltip({
        position: {
            my: "center bottom-10",
            at: "center"
        },
        content: function() {
            return $(this).attr('title');
        }
    });
});

