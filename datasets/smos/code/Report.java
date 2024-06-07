package smos.bean;

import java.util.ArrayList;

import smos.exception.InvalidValueException;

/**
 * 
 * Class used to model a report card.
 * 
 * @author Luigi Colangelo 
 * @version 1.0
 * 
 *          2009 � Copyright by SMOS
 */
public class Report {
	public ArrayList<Votes> pagella; 
	
	/**
	 * The class manufacturer
	 */
	public Report(){
		pagella=new ArrayList<Votes>();
	}
	
	/**
	 * Method that returns the vote from the index in the input array
	 * @param pInd rating in array
	 * @return the rating of the given index
	 * @throws InvalidValueException
	 */
	public Votes getVotes(int pInd) throws InvalidValueException{
		if(pInd<0 || pInd>=pagella.size())throw new InvalidValueException("indice non valido!");
		return (pagella.get(pInd));
	}
	
	/**
	 * Method that adds a vote to the array.
	 * @param pVotes the vote to add
	 * @throws InvalidValueException
	 */
	public void addVotes(Votes pVotes) throws InvalidValueException{
		if(pVotes==null)throw new InvalidValueException("invalid vote!");
		else pagella.add(pVotes);
	}
	
	/**
	 * Method to delete a vote from the array
	 * @param pId is the index of the vote to be deleted from the array.
	 * @throws InvalidValueException 
	 */
	public void remove(int pId) throws InvalidValueException{
		if(pId<0 || pId>=pagella.size())throw new InvalidValueException("invalid index!");
		pagella.remove(pId);
	}
	
	
	public String ToString(){
		String pag="";
		for(Votes e: pagella){
			pag=pag+"\n"+e.toString();
		}
	return pag;
	}
    
}
