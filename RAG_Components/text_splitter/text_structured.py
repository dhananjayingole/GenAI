# it is a text structured based Splitting.
# it chunks the sentences based on paragraph first, then based on chunk size.
# until it become less than or equal to the chunk_size.
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Cricket is a bat-and-ball sport played between two teams of eleven players. It is known for its rich history, 
strategic depth, and varying formats that range from matches lasting five days to high-energy games finishing in just three hours.
Here is a detailed breakdown of the sport.1. The ObjectiveThe core objective of cricket is simple: score more runs than the opposing team.One team bats (trying to score runs) while the other fields and bowls (trying to stop runs and get the batters "out").
After a set number of "overs" or when 10 batters are out, the teams switch roles.The team with the highest number of runs at the end wins.2. The Playing AreaCricket is played on a large oval field. In the distinct center is a rectangular strip called the pitch.
The Pitch: It is 22 yards (20.12 meters) long. 
At each end of the pitch are three wooden stakes called stumps topped with two small bails; together, these form the wicket.The Crease:
 White lines painted on the pitch that define the "safe zones" for the batter and the limit line for the bowler.3. Gameplay MechanicsBowlingThe fielding team has a designated "bowler" who delivers the ball from one end of the pitch to the other, 
attempting to hit the wicket. An "Over" consists of 6 legal deliveries.
 After 6 balls, a different bowler takes over from the opposite end.Batting & Scoring RunsTwo batters are on the pitch at a time. The "striker" faces the bowler, and the "non-striker" stands at the bowler's end.Running: If the batter hits the ball, both batters can run to the opposite end of the pitch. 
 Each successful exchange counts as 1 Run.Boundaries: If the ball reaches the boundary rope:4 Runs: If the ball touches the ground before crossing the rope.6 Runs:
 If the ball flies over the rope without touching the ground.Dismissals (Getting "Out")A batter can be dismissed in 10 ways, but these are the most common:Bowled: The ball hits the stumps and dislodges the bails.Caught: A fielder catches the ball after the batter hits it, before it touches the ground.LBW (Leg Before Wicket): 
 The ball hits the batter's leg in line with the stumps, preventing it from hitting the wicket.Run Out: A fielder hits the wickets with the ball while the batters are running and are not yet in their safe ground (crease).Stumped: The wicketkeeper puts down the wicket while the batter is out of their crease and not attempting a run.4. The Three Major FormatsCricket is unique because it is played in three distinct formats,
each requiring different strategies.FormatDurationClothingDescriptionTest CricketUp to 5 DaysWhiteThe oldest and most prestigious format. It tests endurance and technique. Each team has two innings (bats twice). The match can end in a Draw if time runs out.ODI (One Day International)~8 HoursColoredEach team gets 50 overs. It is a balance of strategy and aggression. The World Cup is played in this format.T20 (Twenty20)~3 HoursColoredEach team gets only 20 overs. It is fast-paced, explosive, and favors aggressive hitting. Leagues like the IPL use this format.5. Fielding PositionsCricket has some of the most complex fielding positions in sports. The captain places fielders based on the bowler's strategy.Slip/Gully: Behind the batter to catch edges (common in Test cricket).
 Cover/Point: On the "off-side" (the side the batter faces) to stop drives.Mid-on/Mid-off: Straight down the ground to stop powerful straight shots.Deep/Long: Fielders placed on the boundary rope to stop 4s and 6s.6. Major TournamentsICC Cricket World Cup (ODI): Held every 4 years; considered the pinnacle of the limited-overs game.ICC T20 World Cup: The premier tournament 
for the shortest format.The Ashes: A historic Test series played between England and Australia (dating back to 1882).IPL (Indian Premier League): 
The most famous domestic T20 league, featuring top players from around the world.7. Unique TerminologyDuck: Getting out for 0 runs.Century: A single batter scoring 100 runs in one innings.Maiden Over: An over where the bowler concedes 0 runs.Hat-trick:
A bowler taking 3 wickets in 3 consecutive balls.Howzat?: The cry fielders shout (appealing) to the umpire when they think a batter is out"""

# initised the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
)

# perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)