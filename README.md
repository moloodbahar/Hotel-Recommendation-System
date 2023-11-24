# Hotel Recommendation System
This project is a Context-Aware Implicit Feedback based Hotel Recommender System for Anonymous Business Travellers who buy a business flight ticket. This project is part of my master thesis project.

## Problem Statement
For Business travelers, there is a travel policy among companies, travel agencies and Hotels. These travel policies are unknown to flight provider companies and they did not consider it in Hotel Optimizer. There are not any standards for these travel policies because each TMC defines their own policies to work with different companies and with different hotel chains in different cities. The main step for providing good recommendations for business travelers should be understanding these unknown business policies. 

Traditional recommender systems operates in the two-dimensional User × Item space; because of that, they make their recommendations based only on the
user and item information and do not take into the consideration additional contextual information that may be crucial in some applications. However, in many situations the utility of a certain product/service to a user may depend significantly on other parameters like time (e.g., the time of the year, such as season or month, or the day of the week). It may also depend on the person with whom the product will be consumed or shared and under which circumstances. In such situations it may not be sufficient to simply recommend items to users; the recommender system must take additional contextual information, such as time, place into consideration when recommending a product [Adomavicius 2005b]. For example, when recommending hotels to a traveler, the system should also consider the place of those hotels to be in the same city as destination city of the traveler, budget and time can be other contexts which can be considered as traveling conditions and restrictions and many other contextual information. Here, for each traveler, the relation of the company, travel agency and the hotels which has the contract with that travel agency should be taken into account. It is important to extend traditional two dimensional User × Item recommendation methods to multidimensional settings. In addition, in several researches, it was shown that considering knowledge about users into the recommendation algorithm in certain applications can lead to better recommendations [Herlocker 2000]. To that end, this thesis focuses to design a hotel recommendation system for the anonymous business travelers when they reserve a flight. They are anonymous because even in the booking data, the company which booked these hotels are not mentioned. Also, gathering information about companies without their permission is not allowed. Then, the first step should be applying a method to find the related company to each transaction. Since our target group is business travelers, we should consider a group of customers from the same company as one user. Then actually for finding a user, we have to cluster transactions from the same company in one group.

Also, the only feedback which we have about interest of companies to book hotels are derived from historical data of booking and we do not have access to the direct ratings of users about hotels. Then we should use the methods which have a better performance for implicit feedback data.


## Objectives
The final approach which we implement to reach these goals for this project guides us to these deliverables:
  • Developing a model-based recommendation system which can work just based on implicit data;
  • Creating the profiles per each user and item by just using the implicit data and the interaction of the users with items based on the transactions data;
  • Having a better performance by embedding these created profiles in the learning process of model-based recommender system;
  • Considering context to give more accurate recommendations to each customer

To know more about this project look at here : https://armanmolood.wixsite.com/moloodarman/post/context-aware-implicit-feedbackbased-hotel-recommender-system-for-anonymous-business-travellers

