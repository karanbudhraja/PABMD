import heatbugs.HeatBugs;
import sim.engine.*;

public class run_heatbugs extends SimState
{
	public run_heatbugs(long seed)
	{
		super(seed);	
	}

	public static void main(String[] args)
	{
		// initialize
		SimState state = new HeatBugs(System.currentTimeMillis());
		
		// set parameters
		HeatBugs heatBugs = (HeatBugs)state;
		heatBugs.setDiffusionConstant(0.0);

		// run
		state.start();

		do
		{
			if (!state.schedule.step(state)) 
				break;
		}
		while(state.schedule.getSteps() < 5000);
		
		// save state as image
		// get some state information
		System.out.println(heatBugs.getRandomMovementProbability());

		state.finish();
		System.exit(0);
	}
}
