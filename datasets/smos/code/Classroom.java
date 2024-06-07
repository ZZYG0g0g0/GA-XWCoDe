package smos.bean;

import java.io.Serializable;

public class Classroom implements Serializable{

	/**
	 * Class used to model a class
	 * @author Nicola Pisanti
	 * @version 1.0
	 */
	
	private static final long serialVersionUID = -8295647317972301446L; 

	private int idClassroom; //Class Id
	private int idAddress;	//Address ID
	private String name;	//Class name
	private int academicYear; //Academic year of the class, to include the year of the first semester.
	
	
	public Classroom(){
		this.idAddress=0;
		this.idClassroom=0;
		
	}
	
	
	/**
	 * Method returning the academic year
	 * @return A whole representing the school year of the first semester of the class.
	 */
	public int getAcademicYear() {
		return academicYear;
	}
	
	
	
	/**
	 * Method that sets the academic year
	 * @param The new academic year to be set
	 */
	public void setAcademicYear(int pAcademicYear) {
		this.academicYear = pAcademicYear;
	}
	
	
	/**
	 * Method to get the class address ID
	 * @return an integer representing the class address ID
	 */
	public int getIdAddress() {
		return idAddress;
	}
	
	
	/**
	 * Method setting class address ID
	 * @param The new ID to set
	 */
	public void setIdAddress(int pIdAddress) {
		this.idAddress = pIdAddress;
	}
	
	
	/**
	 * Method that returns the class ID
	 * @return an integer representing the class ID
	 */
	public int getIdClassroom() {
		return idClassroom;
	}
	
	
	/**
	 * Method setting class ID
	 * @param The new ID to set
	 */
	public void setIdClassroom(int pIdClassroom) {
		this.idClassroom = pIdClassroom;
	}
	
	
	/**
	 * Method that returns the class name
	 * @return a string representing the class name
	 */
	public String getName() {
		return name;
	}
	
	
	/**
	 * Method setting class name
	 * @param The new name to set
	 */
	public void setName(String pName) {
		this.name = pName;
	}
	
	
	
	public String toString(){
		
		return (name + " "+ academicYear+ " ID: "+ idClassroom);
	}
	
	
}
