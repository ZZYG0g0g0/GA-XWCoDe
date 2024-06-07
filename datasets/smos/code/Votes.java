package smos.bean;

import java.io.Serializable;

import smos.exception.InvalidValueException;

/**
 * 
 * Class used to model a grade.
 * 
 * @author Luigi Colangelo
 * @version 1.0
 * 
 *          2009 ?Copyright by SMOS
 */
public class Votes implements Serializable {

	/**
	 * Class used to model a grade
	 * 
	 */
	private static final long serialVersionUID = 3014235634635608576L;
    private int id_votes;
    private int id_user;
    private int teaching;
    private int written;
    private int oral;
    private int laboratory;
    private int accademicYear;
    private int turn;
    
    /**
     * The constructor of the class
     */
    public Votes(){
    	id_votes=0;
    }
    
    /**
     * Method that returns the id of the vote.
     * @return the id of the vote
     */
	public int getId_votes() {
		return id_votes;
	}
	
	/**
	 * Method that sets the id of the vote
	 * @param pId_votes
     *             the id to set
	 */
	public void setId_votes(int pId_votes) {
		this.id_votes = pId_votes;
	}
	
	/**
	 * Method that returns the id of the user connected to the vote
	 * @return the user's id
	 */
	public int getId_user() {
		return id_user;
	}
	
	/**
	 * Method that sets the user's id related to the vote
	 * @param pId_user
	 *               the id to set
	 */
	public void setId_user(int pId_user) {
		this.id_user = pId_user;
	}
	
	/**
	 * method that returns the code of teaching the vote
	 * @return the teaching method
	 */
	public int getTeaching() {
		return teaching;
	}
	
	/**
	 * Method that sets the teaching code relating to the grade
	 * @param pTeaching
	 *              the teaching code
	 */
	public void setTeaching(int pTeaching) {
		this.teaching = pTeaching;
	}
	
	/**
	 * Method that returns the grade of the writing
	 * @return the vote in the writing
	 */
	public int getWritten() {
		return written;
	}
	
	/**
	 * Method that sets the grade of the writing, checking that it is between 0 and 10
	 * @param pWritten
	 */
	public void setWritten(int pWritten) throws InvalidValueException {
		if (pWritten < 0 || pWritten > 10)
			throw new InvalidValueException();
		else
		this.written = pWritten;
	}
	
	/**
	 *  method that returns the grade of the oral
	 * @return the oral vote
	 */
	public int getOral() {
		return oral;
	}
	
	/**
	 * Method that sets the grade of the oral, checking that it is between 0 and 10
	 * @param pOral
	 *            the oral vote to be set
	 */
	public void setOral(int pOral) throws InvalidValueException{
		if (pOral < 0 || pOral > 10)
			throw new InvalidValueException();
		else
		this.oral = pOral;
	}
	
	/**
	 * Method that returns the laboratory grade
	 * @return the lab grade
	 */
	public int getLaboratory() {
		return laboratory;
	}
	
	/**
	 * method that sets the laboratory grade, checking that it is between 0 and 10
	 * @param pLaboratory
	 *                 the vote of the laboratory to be set
	 */
	public void setLaboratory(int pLaboratory)throws InvalidValueException {
		if (pLaboratory < 0 || pLaboratory > 10)
			throw new InvalidValueException();
		else
		this.laboratory = pLaboratory;
	}
	
	/**
	 * Method that returns the academic year of the grade
	 * @return the academic year
	 */
	public int getAccademicYear() {
		return accademicYear;
	}
	
	/**
	 * method that sets the academic year of the grade
	 * @param pAccademicYear
	 */
	public void setAccademicYear(int pAccademicYear) {
		this.accademicYear = pAccademicYear;
	}
	
	/**
	 * Method that returns the quarter of the vote
	 * @return the semester of the vote (0 or 1)
	 */
	public int getTurn() {
		return turn;
	}
	
	/**
	 * Method that sets the quarter of the vote
	 * @param pTurn
	 *            the semester of the vote to be set
	 */
	public void setTurn(int pTurn) {
		this.turn = pTurn;
	}
	
	public String toString(){
		return("id voto= "+id_votes+" id user= "+id_user+" id insegnamento= "+teaching+" scritto= "+written+" orale= "+oral+" laboratorio= "+laboratory+" anno accademico= "+accademicYear+" quadrimestre= "+turn);
	}
    
 
}
